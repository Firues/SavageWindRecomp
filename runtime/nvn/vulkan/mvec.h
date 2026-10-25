// Per-object motion vectors (mvec.c): the scene draws of each frame, matched to the same draws of the previous
// frame, with copies of the inputs their vertex shaders read (the game's constant buffers, the instanced streams),
// kept for the frame after. See docs/motion-vectors-spike.md for why it works this way.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "vk.h"

#define MVEC_CB 14    // the game's constant buffers, hardware slots 3..16
#define MVEC_INST 4   // instanced streams copied per draw
// A motion program's inputs (maxwell/translate.c, ShaderKey mvec): the previous frame's history buffer at this binding
// of the vertex set, and push constants: uint pb[14] (where each constant buffer slot 3..16 starts, in 16-byte units)
// then uint pbn[14] (its size, in 16-byte units)
#define MVEC_HISTORY_BINDING 63
#define MVEC_PUSH 112

typedef struct { uint64_t addr, size; uint32_t stride, divisor; } MvecStream;

// A scene draw as draw_ sees it (exec.c): what identifies it and where its vertex shader's inputs are in guest memory
typedef struct {
    uint64_t frame;  // presented frames so far (renderer_present_count)
    uint64_t gen;    // guest copy generation (exec.c gcopy_gen): equal ranges of one generation hold equal bytes
    uint64_t vs, fs;  // shader code addresses
    uint64_t iaddr;   // index buffer (0: not indexed)
    uint32_t count, first, inst, base_inst;
    int32_t base_vertex;
    int nstream;
    MvecStream stream[16];
    uint32_t vs_streams;  // streams the vertex shader's attributes read
    int ncb;
    struct { uint8_t slot; uint64_t addr, size; } cb[MVEC_CB];
} MvecDrawIn;

// One recorded draw: its identity, its match in the previous frame and its copied inputs (offsets in its frame's
// history buffer, mvec_records)
typedef struct {
    uint64_t key;
    uint32_t ord;   // among this frame's draws with the same key, in draw order
    int32_t prev;   // the matching record of the previous frame, -1: none
    uint8_t ncb, ninst;
    uint8_t cb_slot[MVEC_CB];
    uint32_t cb_off[MVEC_CB], cb_size[MVEC_CB];
    uint8_t inst_stream[MVEC_INST];
    uint32_t inst_off[MVEC_INST], inst_size[MVEC_INST];
} MvecRec;

bool mvec_enabled(void);  // SAVAGEWIND_MVEC=1
// Records a scene draw of the current frame: its index among the frame's records, -1 when not recorded (off, or the
// history buffer is full: grown for the next frame).
int mvec_record(const MvecDrawIn *d);
// The current and the previous frame's records and history buffers (NULL / 0 when there is no previous frame)
const MvecRec *mvec_records(bool previous, int *n, VkBuffer *history);
// The push constants of the current frame's record `rec` (mvec_record's index): its match's constant buffer copies in
// the previous frame's history. false: no match, or the match lacks a buffer this draw reads (no motion program).
bool mvec_push(int rec, uint32_t pc[MVEC_PUSH / 4]);
void mvec_drawn(void);  // a recorded draw was drawn with its motion program (statistics)
// The previous frame's history buffer, for MVEC_HISTORY_BINDING (any valid buffer before there is one)
VkDescriptorBufferInfo mvec_history_info(void);
// SAVAGEWIND_MVEC_VIEW=<pixels>: the vectors drawn over the scene (exec.c mvec_scene_view), full colour at that length
// (default 16; negative: raw, red = written, green / blue = |x| / |y| NDC times its size); 0 when off
float mvec_view_enabled(void);
