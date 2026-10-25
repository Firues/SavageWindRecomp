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
// History buffers: MVEC_SLOTS host-visible buffers, one per frame in flight and the frame being recorded, each
// holding one frame's copies; a frame's copies are read again during the next frame (as "previous"), so a slot is
// reused only once the GPU is done with the frames that read it. A frame that fills its slot stops recording (its
// remaining draws get no motion) and the slot is grown before its next use.
// SAVAGEWIND_MVEC=1: on (off by default while this is being built). SAVAGEWIND_MVEC_LOG=<frames>: statistics every
// that many frames (default 240).
#include <stdlib.h>
#include <string.h>

#include "../../rt.h"
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
} Slot;

static Slot slots[MVEC_SLOTS];
static int cur = -1, prev = -1;  // slots of the frame being recorded and of the one before it
static uint64_t cur_frame = ~0ull;
static Dedupe dedupe[DEDUPE];
static Index prev_index[INDEX], ord_index[INDEX];

static struct {  // since the last log line
    uint64_t frames, draws, matched, repeated, unmatched, dropped, cb_bytes, inst_bytes, copied, dedupe_hits, max_used;
} stat;

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
    LOG("[mvec] frame %llu: %.0f scene draws a frame, matched %.1f%% (%.0f with a repeated key), unmatched %.0f, "
        "not recorded %.0f; inputs %.0f KB constants + %.0f KB instance data a frame, %.0f KB copied (%.0f copies "
        "shared); history slot at most %.1f MB used of %.1f\n",
        (unsigned long long)frame, stat.draws / f, stat.draws ? 100.0 * stat.matched / stat.draws : 0.0, stat.repeated / f,
        stat.unmatched / f, stat.dropped / f, stat.cb_bytes / f / 1024, stat.inst_bytes / f / 1024, stat.copied / f / 1024,
        stat.dedupe_hits / f, stat.max_used / 1048576.0, cur >= 0 ? slots[cur].cap / 1048576.0 : 0.0);
    memset(&stat, 0, sizeof stat);
}

// A new presented frame: the slot it records into, the previous frame's index
static void frame_begin(uint64_t frame) {
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
        r->cb_slot[r->ncb] = d->cb[k].slot;
        r->cb_off[r->ncb] = off;
        r->cb_size[r->ncb++] = (uint32_t)d->cb[k].size;
        stat.cb_bytes += d->cb[k].size;
    }
    r->key = key;
    r->ord = next_ordinal(key);
    r->prev = prev_lookup(key, r->ord);
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

const MvecRec *mvec_records(bool previous, int *n, VkBuffer *history) {
    int i = previous ? prev : cur;
    if (i < 0) { *n = 0; *history = VK_NULL_HANDLE; return NULL; }
    *n = slots[i].nrec;
    *history = slots[i].buf;
    return slots[i].rec;
}
