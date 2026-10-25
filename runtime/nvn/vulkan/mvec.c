// Per-object motion vectors, step 1: draw identity and the previous frame's inputs.
//
// BOTW writes its constants into fresh memory every frame (three quarters of the constant buffers that move between
// frames carry the same bytes), so "the previous frame's data at this address" doesn't exist. Instead each scene draw
// (targets at the internal resolution, RS_SCENE) is identified by a key - its shaders, the non-instanced vertex streams
// its vertex shader reads, its index range - and an ordinal among the frame's draws with that key, and matched to the
// draw of the previous frame with the same key and ordinal (93-99% of them in the spike). The bytes its vertex shader
// reads that change between frames are copied, at draw time, into the frame's history buffer: the game's constant
// buffers (bones and per-object matrices included: no scene vertex shader reads global memory) and the instanced
// streams (rewritten every frame, in place). Per-vertex streams are not: the game never rewrites them between frames.
//
// History buffers (write-combined, often across the PCIe BAR: the CPU writes them and must never read them back, which
// took 17 us a draw): MVEC_SLOTS host-visible buffers, one per frame in flight and the frame being recorded, each
// holding one frame's copies; a frame's copies are read again during the next frame (as "previous"), so a slot is
// reused only once the GPU is done with the frames that read it. A frame that fills its slot stops recording (its
// remaining draws get no motion) and the slot is grown before its next use.
// SAVAGEWIND_MVEC=1: on (off by default while this is being built). SAVAGEWIND_MVEC_LOG=<frames>: statistics every
// that many frames (default 240).
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../rt.h"
#include "../output_scale.h"
#include "mvec.h"

#define MVEC_SLOTS 5
#define HIST_START (16ull << 20)
#define DEDUPE 8192  // copies of one (address, size, generation) shared within a frame
#define INDEX 32768  // (key, ordinal) -> record of the previous frame; key -> draws so far in the current frame

typedef struct { uint64_t addr, size, gen; uint32_t off; } Dedupe;
typedef struct { uint64_t key; uint32_t ord; int32_t rec; } Index;  // (rec -1: empty)

typedef struct {
    VkBuffer buf;
    VmaAllocation mem;
    uint8_t *ptr;
    uint64_t cap, used, grow, seq;  // seq: the last submission that may read it
    uint64_t frame;                 // the frame it holds (~0: none)
    MvecRec *rec;
    int nrec, cap_rec;
    bool full;
    MvecCamera cam;  // (camera_finish, once the frame is recorded)
} Slot;

static Slot slots[MVEC_SLOTS];
static int cur = -1, prev = -1;  // slots of the frame being recorded and of the one before it
static uint64_t cur_frame = ~0ull;
static Dedupe dedupe[DEDUPE];
static Index prev_index[INDEX], ord_index[INDEX];
// The previous frame's records sorted by key (pkeys: their keys, pidx: their indices), and which are taken
static uint64_t *pkeys;
static int32_t *pidx;
static bool *ptaken;
static int npk, cap_pk;

static struct {  // since the last log line
    uint64_t dist_hist[2][6];  // chosen matches' fingerprint distances: [key with one candidate / several][bucket]
    uint64_t not_ordinal;      // matches the draw order alone would have made differently
    // draws with the scene shaders' layout (camera block at slot 3, the object's 3x4 world matrix at slot 4) whose object
    // moved more than 1 cm since its match, matched by fingerprint and as draw order alone would have: mismatches look
    // like jumps (most of the world stands still)
    uint64_t moved_checked, moved_order, moved_fp;
    uint64_t frames, draws, matched, repeated, unmatched, dropped, cb_bytes, inst_bytes, copied, dedupe_hits, max_used, motion,
        cameras;    // frames with a valid camera (camera_finish)
} stat;
// The frame's slot-3 copies by guest address (the camera block: the one most draws read, camera_finish)
#define CAMC 32
static struct { uint64_t addr; uint32_t off, size, n; } camc[CAMC];
static int ncamc;

static void mat4_mul(const float *a, const float *b, float *o) {  // row-major: o = a * b
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) o[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
}

// Inverse of a 4x4 (row-major), by cofactors; false when singular
static bool mat4_inverse(const float *m, float *o) {
    float a[16];
    a[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    a[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    a[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    a[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    a[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    a[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    a[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    a[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    a[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    a[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    a[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    a[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    a[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    a[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    a[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    a[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    double det = (double)m[0] * a[0] + (double)m[1] * a[4] + (double)m[2] * a[8] + (double)m[3] * a[12];
    if (fabs(det) < 1e-30) return false;
    for (int i = 0; i < 16; i++) o[i] = (float)(a[i] / det);
    return true;
}

// Whether a camera block's view and projection look like the world camera: a rigid view, and a perspective projection
// with the world camera's field of view and aspect (rt_world_projection; 2%: the field of view can ease between the
// frame that made the block and the last matrix) and (0, 0, -1, 0) as its last row. Its other entries are left free:
// the near plane moves, and an offset (the jitter) may be in [0][2] / [1][2].
static bool camera_plausible(const MvecCamera *c) {
    for (int r = 0; r < 3; r++) {
        const float *a = &c->view[r * 4];
        if (fabsf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] - 1) > 0.01f) return false;
        for (int q = r + 1; q < 3; q++) {
            const float *b = &c->view[q * 4];
            if (fabsf(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) > 0.01f) return false;
        }
    }
    const float *p = c->proj;
    if (p[12] != 0 || p[13] != 0 || p[14] != -1 || p[15] != 0 || p[1] != 0 || p[4] != 0) return false;
    float w[4];
    rt_world_projection(w);
    if (w[0] <= 0 || w[1] <= 0) return false;
    return fabsf(p[5] * w[0] - 1) < 0.02f && fabsf(p[0] * w[0] * w[1] - 1) < 0.02f;
}

// A frame's camera, from its camera block's copy in the slot's history (the block most of the frame's draws so far
// read); false when there is none or it fails camera_plausible
static bool camera_compute(Slot *s, MvecCamera *c) {
    c->valid = false;
    int best = -1;
    for (int i = 0; i < ncamc; i++) if (best < 0 || camc[i].n > camc[best].n) best = i;
    if (best < 0 || camc[best].size < 11 * 16) return false;
    float rows[11 * 4];
    memcpy(rows, s->ptr + camc[best].off, sizeof rows);  // (write-combined memory: one read of 176 bytes)
    memcpy(c->view, rows, 12 * 4);
    memcpy(c->proj, rows + 7 * 4, 16 * 4);
    float v4[16];
    memcpy(v4, c->view, 12 * 4);
    v4[12] = v4[13] = v4[14] = 0, v4[15] = 1;
    mat4_mul(c->proj, v4, c->vp);
    static bool warned;
    if (!camera_plausible(c) || !mat4_inverse(c->vp, c->inv_vp)) {
        if (!warned) LOG("[mvec] frame %llu: the scene camera block (%llx, %u draws) is not the world camera's view and "
                         "projection: no camera vectors\n", (unsigned long long)s->frame, (unsigned long long)camc[best].addr,
                         camc[best].n);
        warned = true;
        return false;
    }
    c->valid = true;
    return true;
}

// The finished frame's camera (when the next frame begins)
static void camera_finish(Slot *s) {
    if (camera_compute(s, &s->cam)) stat.cameras++;
    ncamc = 0;
}


bool mvec_enabled(void) {
    static int on = -1;
    if (on < 0) on = getenv("SAVAGEWIND_MVEC") && atoi(getenv("SAVAGEWIND_MVEC")) != 0;
    return on;
}

static uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h * 0xff51afd7ed558ccdull;
}

static uint32_t slot_hash(uint64_t key, uint32_t ord) { return (uint32_t)(mix(key, ord) >> 32) & (INDEX - 1); }

static void slot_alloc(Slot *s, uint64_t cap) {
    if (s->buf) {
        vk_wait_seq(s->seq);
        vmaDestroyBuffer(g_vk.vma, s->buf, s->mem);
    }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &s->buf, &s->mem, &info));
    s->ptr = info.pMappedData;
    s->cap = cap;
}

static void log_stats(uint64_t frame) {
    static int every = -1;
    if (every < 0) every = getenv("SAVAGEWIND_MVEC_LOG") ? atoi(getenv("SAVAGEWIND_MVEC_LOG")) : 240;
    if (every <= 0 || ++stat.frames < (uint64_t)every) return;
    double f = (double)stat.frames;
    LOG("[mvec] frame %llu: %.0f scene draws a frame, matched %.1f%% (%.0f with a repeated key, %.1f%% drawn with motion), "
        "unmatched %.0f, "
        "not recorded %.0f; inputs %.0f KB constants + %.0f KB instance data a frame, %.0f KB copied (%.0f copies "
        "shared); history slot at most %.1f MB used of %.1f; camera found in %.0f%% of the frames\n",
        (unsigned long long)frame, stat.draws / f, stat.draws ? 100.0 * stat.matched / stat.draws : 0.0, stat.repeated / f,
        stat.draws ? 100.0 * stat.motion / stat.draws : 0.0, stat.unmatched / f, stat.dropped / f, stat.cb_bytes / f / 1024, stat.inst_bytes / f / 1024, stat.copied / f / 1024,
        stat.dedupe_hits / f, stat.max_used / 1048576.0, cur >= 0 ? slots[cur].cap / 1048576.0 : 0.0,
        100.0 * stat.cameras / f);
    for (int g = 0; g < 2; g++)
        LOG("[mvec]   fingerprint distances, keys with %s: %.0f < 1e-4, %.0f < 1e-3, %.0f < 0.01, %.0f < 0.05, %.0f < 0.2, "
            "%.0f more%s", g ? "several candidates" : "one candidate", stat.dist_hist[g][0] / f, stat.dist_hist[g][1] / f,
            stat.dist_hist[g][2] / f, stat.dist_hist[g][3] / f, stat.dist_hist[g][4] / f, stat.dist_hist[g][5] / f,
            g ? "" : "\n");
    LOG("; %.0f a frame matched differently than by draw order\n", stat.not_ordinal / f);
    LOG("[mvec]   of %.0f draws with a world matrix, moved > 1 cm: %.1f matched by fingerprint (%.1f by draw order)\n",
        stat.moved_checked / f, stat.moved_fp / f, stat.moved_order / f);
    memset(&stat, 0, sizeof stat);
}

static const Slot *sort_prev;  // (cmp_by_key's records)
static int cmp_by_key(const void *a, const void *b) {
    const MvecRec *x = &sort_prev->rec[*(const int32_t *)a], *y = &sort_prev->rec[*(const int32_t *)b];
    if (x->key != y->key) return x->key < y->key ? -1 : 1;
    return *(const int32_t *)a < *(const int32_t *)b ? -1 : *(const int32_t *)a > *(const int32_t *)b;
}

static int32_t prev_match(MvecRec *r);

// A new presented frame: the slot it records into, the previous frame's index
static void frame_begin(uint64_t frame) {
    if (cur >= 0) camera_finish(&slots[cur]);
    if (cur >= 0) {
        if (slots[cur].used > stat.max_used) stat.max_used = slots[cur].used;
        log_stats(frame);
    }
    int last = cur;
    cur = (int)(frame % MVEC_SLOTS);
    cur_frame = frame;
    Slot *s = &slots[cur];
    vk_wait_seq(s->seq);  // (MVEC_SLOTS frames ago: long done in practice)
    if (!s->buf || s->grow > s->cap) slot_alloc(s, s->grow > HIST_START ? s->grow : HIST_START);
    s->used = 0;
    s->nrec = 0;
    s->full = false;
    s->frame = frame;
    prev = last >= 0 && slots[last].frame + 1 == frame ? last : -1;
    memset(dedupe, 0, sizeof dedupe);
    for (int i = 0; i < INDEX; i++) prev_index[i].rec = ord_index[i].rec = -1;
    if (prev < 0) return;
    for (int i = 0; i < slots[prev].nrec; i++) {  // (key, ordinal) -> record
        const MvecRec *r = &slots[prev].rec[i];
        uint32_t h = slot_hash(r->key, r->ord);
        while (prev_index[h].rec >= 0) h = (h + 1) & (INDEX - 1);
        prev_index[h] = (Index){r->key, r->ord, i};
    }
    npk = slots[prev].nrec;  // (by key: each key's candidates together, in draw order)
    if (npk > cap_pk) {
        cap_pk = npk * 2;
        pkeys = realloc(pkeys, (size_t)cap_pk * sizeof *pkeys);
        pidx = realloc(pidx, (size_t)cap_pk * sizeof *pidx);
        ptaken = realloc(ptaken, (size_t)cap_pk * sizeof *ptaken);
    }
    for (int i = 0; i < npk; i++) pidx[i] = i;
    sort_prev = &slots[prev];
    qsort(pidx, (size_t)npk, sizeof *pidx, cmp_by_key);
    for (int i = 0; i < npk; i++) pkeys[i] = slots[prev].rec[pidx[i]].key;
    memset(ptaken, 0, (size_t)npk * sizeof *ptaken);
}

// How far apart two fingerprints are: the mean relative difference of their values (0 equal, 1 unrelated); values
// that are no finite floats (integers, flags) count as equal or different by their bits
static float fingerprint_distance(const MvecRec *a, const MvecRec *b) {
    int n = a->nfeat < b->nfeat ? a->nfeat : b->nfeat;
    if (!n || a->nfeat != b->nfeat) return 1;
    float d = 0;
    for (int i = 0; i < n; i++) {
        float x = a->feat[i], y = b->feat[i];
        if (!isfinite(x) || !isfinite(y)) { d += memcmp(&x, &y, 4) ? 1.0f : 0.0f; continue; }
        float s = fabsf(x) + fabsf(y);
        d += s > 1e-6f ? fabsf(x - y) / s : 0.0f;
    }
    return d / (float)n;
}

// The previous frame's record for a draw: among the ones with its key and not taken yet, the nearest fingerprint.
// Culling shifts a key's draws by a few places, not hundreds: the search starts at the draw's own place in the key's
// order and looks MATCH_WINDOW places either way (a key can have hundreds of draws: comparing all of them took ~40 us a
// draw), and stops at an identical fingerprint (a still object, the most common case).
#define MATCH_WINDOW 24
static int32_t prev_match(MvecRec *r) {
    if (prev < 0 || !npk) return -1;
    int lo = 0, hi = npk;
    while (lo < hi) { int m = (lo + hi) / 2; if (pkeys[m] < r->key) lo = m + 1; else hi = m; }
    int end = lo, ehi = npk;  // (the key's last candidate + 1)
    while (end < ehi) { int m = (end + ehi) / 2; if (pkeys[m] <= r->key) end = m + 1; else ehi = m; }
    int ncand = end - lo, at = lo + (int)r->ord;
    if (!ncand) return -1;
    if (at >= end) at = end - 1;
    int best = -1;
    float bd = 2;
    for (int k = 0; k <= 2 * MATCH_WINDOW && bd > 0; k++) {  // (at, at + 1, at - 1, at + 2, ...)
        int i = at + (k & 1 ? (k + 1) / 2 : -(k / 2));
        if (i < lo || i >= end || ptaken[i]) continue;
        float d = fingerprint_distance(r, &slots[prev].rec[pidx[i]]);
        if (d < bd) bd = d, best = i;
    }
    if (best < 0) return -1;
    static float limit = -1;  // SAVAGEWIND_MVEC_MATCH_LIMIT: the largest fingerprint distance of a match (default 0.2)
    if (limit < 0) limit = getenv("SAVAGEWIND_MVEC_MATCH_LIMIT") ? (float)atof(getenv("SAVAGEWIND_MVEC_MATCH_LIMIT")) : 0.2f;
    if (bd >= limit) { stat.dist_hist[ncand > 1][5]++; return -1; }  // (a new object: the camera's vector)
    ptaken[best] = true;
    r->dist = bd;
    static const float edges[5] = {1e-4f, 1e-3f, 0.01f, 0.05f, 0.2f};
    int bucket = 0;
    while (bucket < 5 && bd >= edges[bucket]) bucket++;
    stat.dist_hist[ncand > 1][bucket]++;
    return pidx[best];
}

static int32_t prev_lookup(uint64_t key, uint32_t ord) {
    if (prev < 0) return -1;
    for (uint32_t h = slot_hash(key, ord);; h = (h + 1) & (INDEX - 1)) {
        if (prev_index[h].rec < 0) return -1;
        if (prev_index[h].key == key && prev_index[h].ord == ord) return prev_index[h].rec;
    }
}

// The ordinal of this draw among the frame's draws with its key (counts kept in ord_index, ord = count so far)
static uint32_t next_ordinal(uint64_t key) {
    for (uint32_t h = slot_hash(key, 0);; h = (h + 1) & (INDEX - 1)) {
        if (ord_index[h].rec < 0) { ord_index[h] = (Index){key, 1, 0}; return 0; }
        if (ord_index[h].key == key) return ord_index[h].ord++;
    }
}

// Copies [addr, addr + size) of guest memory into the current slot (shared with an earlier copy of the same range in
// the same generation); UINT32_MAX when the slot is full
static uint32_t copy_in(uint64_t addr, uint64_t size, uint64_t gen, bool share) {
    Slot *s = &slots[cur];
    uint32_t h = (uint32_t)(mix(mix(addr, size), gen) >> 40) & (DEDUPE - 1);
    if (share && dedupe[h].addr == addr && dedupe[h].size == size && dedupe[h].gen == gen) {
        stat.dedupe_hits++;
        return dedupe[h].off;
    }
    uint64_t align = g_vk.props.limits.minUniformBufferOffsetAlignment;
    if (align < 16) align = 16;
    uint64_t off = (s->used + align - 1) & ~(align - 1);
    if (off + size > s->cap) {
        s->full = true;
        uint64_t want = (off + size) * 2;
        if (want > s->grow) s->grow = want;
        return UINT32_MAX;
    }
    memcpy(s->ptr + off, (const void *)(uintptr_t)addr, size);
    stat.copied += size;
    s->used = off + size;
    if (share) dedupe[h] = (Dedupe){addr, size, gen, (uint32_t)off};
    return (uint32_t)off;
}

int mvec_record(const MvecDrawIn *d) {
    if (!mvec_enabled()) return -1;
    if (d->frame != cur_frame) frame_begin(d->frame);
    Slot *s = &slots[cur];
    stat.draws++;
    if (s->full) { stat.dropped++; return -1; }
    if (s->nrec == s->cap_rec) {
        s->cap_rec = s->cap_rec ? s->cap_rec * 2 : 4096;
        s->rec = realloc(s->rec, (size_t)s->cap_rec * sizeof *s->rec);
    }
    MvecRec *r = &s->rec[s->nrec];
    memset(r, 0, sizeof *r);
    uint64_t key = mix(mix(mix(d->vs, d->fs), d->iaddr), ((uint64_t)d->count << 32) | d->first);
    key = mix(key, (uint32_t)d->base_vertex);
    uint64_t used_before = s->used;
    for (int i = 0; i < d->nstream && i < 16; i++) {
        const MvecStream *m = &d->stream[i];
        if (!((d->vs_streams >> i) & 1) || !m->addr) continue;
        if (!m->divisor) { key = mix(key, m->addr); continue; }
        if (r->ninst == MVEC_INST) continue;
        uint64_t n = (d->base_inst + d->inst + m->divisor - 1) / m->divisor, size = n * m->stride;
        if (size > m->size && m->size) size = m->size;
        if (!size || size > (8u << 20)) continue;
        uint32_t off = copy_in(m->addr, size, d->gen, false);  // (rewritten in place: never shared across draws)
        if (off == UINT32_MAX) goto full;
        r->inst_stream[r->ninst] = (uint8_t)i;
        r->inst_off[r->ninst] = off;
        r->inst_size[r->ninst++] = (uint32_t)size;
        stat.inst_bytes += size;
    }
    for (int k = 0; k < d->ncb && r->ncb < MVEC_CB; k++) {
        if (!d->cb[k].addr || !d->cb[k].size) continue;
        uint32_t off = copy_in(d->cb[k].addr, d->cb[k].size, d->gen, true);
        if (off == UINT32_MAX) goto full;
        if (d->cb[k].slot == 3) {  // (camera block candidates: camera_finish)
            int i = 0;
            while (i < ncamc && camc[i].addr != d->cb[k].addr) i++;
            if (i == ncamc && ncamc < CAMC) camc[ncamc++] = (typeof(camc[0])){d->cb[k].addr, off, (uint32_t)d->cb[k].size, 0};
            if (i < ncamc) camc[i].n++;
        }
        r->cb_slot[r->ncb] = d->cb[k].slot;
        r->cb_off[r->ncb] = off;
        r->cb_size[r->ncb++] = (uint32_t)d->cb[k].size;
        stat.cb_bytes += d->cb[k].size;
    }
    for (int k = 0; k < d->ncb && r->nfeat + 12 <= MVEC_FEAT; k++) {  // (the fingerprint, from guest memory as it is now)
        if (!d->cb[k].addr || d->cb[k].size < 48) continue;
        memcpy(&r->feat[r->nfeat], (const void *)(uintptr_t)d->cb[k].addr, 48);
        r->nfeat += 12;
    }
    r->key = key;
    r->ord = next_ordinal(key);
    static int by_order = -1;  // SAVAGEWIND_MVEC_MATCH=order: (testing) match by draw order alone
    if (by_order < 0) by_order = getenv("SAVAGEWIND_MVEC_MATCH") && !strcmp(getenv("SAVAGEWIND_MVEC_MATCH"), "order");
    r->prev = by_order ? prev_lookup(key, r->ord) : prev_match(r);
    {  // (statistics) the object's world position (slot 4's translation) against each matching's previous draw
        int32_t po = prev_lookup(key, r->ord), pf = by_order ? -1 : r->prev;
        if (r->nfeat >= 24 && r->cb_slot[0] == 3 && r->cb_slot[1] == 4 && po >= 0 && pf >= 0) {
            stat.moved_checked++;
            const float *t = &r->feat[12];
            for (int m = 0; m < 2; m++) {
                const MvecRec *q = &slots[prev].rec[m ? pf : po];
                if (q->nfeat < 24) continue;
                float dx = t[3] - q->feat[15], dy = t[7] - q->feat[19], dz = t[11] - q->feat[23];
                if (dx * dx + dy * dy + dz * dz > 1e-4f) { if (m) stat.moved_fp++; else stat.moved_order++; }
            }
        }
    }
    if (r->prev >= 0 && r->prev != prev_lookup(key, r->ord)) stat.not_ordinal++;
    if (r->ord) stat.repeated++;
    if (r->prev >= 0) stat.matched++;
    else stat.unmatched++;
    s->seq = vk_pending_seq();
    return s->nrec++;
full:
    s->used = used_before;
    stat.dropped++;
    return -1;
}

void mvec_drawn(void) { stat.motion++; }

bool mvec_push(int rec, uint32_t pc[MVEC_PUSH / 4]) {
    if (cur < 0 || prev < 0 || rec < 0 || rec >= slots[cur].nrec) return false;
    const MvecRec *r = &slots[cur].rec[rec];
    if (r->prev < 0) return false;
    const MvecRec *p = &slots[prev].rec[r->prev];
    memset(pc, 0, MVEC_PUSH);
    for (int k = 0; k < r->ncb; k++) {
        int j = -1;
        for (int m = 0; m < p->ncb; m++) if (p->cb_slot[m] == r->cb_slot[k]) j = m;
        if (j < 0) return false;
        int s = r->cb_slot[k] - 3;
        pc[s] = p->cb_off[j] / 16;
        pc[14 + s] = (p->cb_size[j] + 15) / 16;
    }
    slots[prev].seq = vk_pending_seq();  // (read by this frame's draws too: kept until they are done)
    return true;
}

VkDescriptorBufferInfo mvec_history_info(void) {
    int i = prev >= 0 ? prev : cur;
    return (VkDescriptorBufferInfo){i >= 0 ? slots[i].buf : VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
}

int mvec_camera_only(void) {
    static int on = -1;
    if (on < 0) on = getenv("SAVAGEWIND_MVEC_CAMONLY") ? atoi(getenv("SAVAGEWIND_MVEC_CAMONLY")) : 0;
    return on;
}

float mvec_view_enabled(void) {
    static float px;
    static bool init;
    if (!init) {  // (a negative value: raw, red = written, green / blue = |x| / |y| times its size)
        init = true;
        const char *e = getenv("SAVAGEWIND_MVEC_VIEW");
        px = e ? (atof(e) != 0 ? (float)atof(e) : 16.0f) : 0.0f;
    }
    return px;
}

bool mvec_reprojection(float m[16], float jitter[2]) {
    if (cur < 0 || prev < 0 || !slots[prev].cam.valid) return false;
    MvecCamera now;
    if (!camera_compute(&slots[cur], &now)) return false;
    mat4_mul(slots[prev].cam.vp, now.inv_vp, m);
    // a projection offset (proj[0][2], proj[1][2]) moves the picture by minus itself in NDC
    jitter[0] = -(now.proj[2] - slots[prev].cam.proj[2]);
    jitter[1] = -(now.proj[6] - slots[prev].cam.proj[6]);
    return true;
}

const MvecCamera *mvec_camera(bool previous) {
    int i = previous ? prev : -1;  // (the frame being recorded has no camera yet: the previous one and the one before)
    if (!previous && prev >= 0) i = (prev + MVEC_SLOTS - 1) % MVEC_SLOTS;
    return i >= 0 && slots[i].cam.valid ? &slots[i].cam : NULL;
}

const MvecRec *mvec_records(bool previous, int *n, VkBuffer *history) {
    int i = previous ? prev : cur;
    if (i < 0) { *n = 0; *history = VK_NULL_HANDLE; return NULL; }
    *n = slots[i].nrec;
    *history = slots[i].buf;
    return slots[i].rec;
}
