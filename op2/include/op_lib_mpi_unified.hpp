#pragma once

#include <vector>
#include <utility>
#include <cstdint>

struct PtrBuffers {
    std::vector<void *> gather_ptrs;
    std::vector<void *> scatter_ptrs;

    void **gather_ptrs_d;
    void **scatter_ptrs_d;
};

void *copy_to_device(const void *buf, size_t size);

void set_ptr_buffers(const PtrBuffers *new_ptrs_4, const PtrBuffers *new_ptrs_8);
void realloc_exchange_buffers();

void initiate_gathers();
void wait_gathers();

std::pair<uint32_t *, uint64_t *> get_gather_buffers();
std::pair<uint32_t *, uint64_t *> get_scatter_buffers();

void initiate_scatters();
void wait_scatters();
