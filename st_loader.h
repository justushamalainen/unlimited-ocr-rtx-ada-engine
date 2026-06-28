// Minimal safetensors loader: reads engine/manifest.tsv + mmaps the weight file.
// No JSON dependency; manifest is name<TAB>dtype<TAB>shape<TAB>abs_offset<TAB>nbytes.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;   // pointer into the mmap
    size_t nbytes = 0;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};

class SafeTensors {
public:
    void load(const std::string& manifest_path) {
        std::ifstream mf(manifest_path);
        if (!mf) { fprintf(stderr, "cannot open %s\n", manifest_path.c_str()); exit(1); }
        std::string line, safe_path;
        while (std::getline(mf, line)) {
            if (line.rfind("# safetensors=", 0) == 0) { safe_path = line.substr(14); continue; }
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string name, dtype, shp, off, nb;
            std::getline(ss, name, '\t'); std::getline(ss, dtype, '\t');
            std::getline(ss, shp, '\t');  std::getline(ss, off, '\t'); std::getline(ss, nb, '\t');
            Tensor t; t.dtype = dtype;
            if (!shp.empty()) { std::stringstream s2(shp); std::string d;
                while (std::getline(s2, d, ',')) t.shape.push_back(std::stoll(d)); }
            offsets_[name] = std::stoull(off);
            t.nbytes = std::stoull(nb);
            tensors_[name] = t;
        }
        int fd = open(safe_path.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "cannot open %s\n", safe_path.c_str()); exit(1); }
        struct stat stt; fstat(fd, &stt); filesize_ = stt.st_size;
        base_ = (const uint8_t*)mmap(nullptr, filesize_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base_ == MAP_FAILED) { perror("mmap"); exit(1); }
        close(fd);
        for (auto& kv : tensors_) kv.second.data = base_ + offsets_[kv.first];
    }
    const Tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) { fprintf(stderr, "missing tensor: %s\n", name.c_str()); exit(1); }
        return it->second;
    }
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    size_t count() const { return tensors_.size(); }
private:
    const uint8_t* base_ = nullptr;
    size_t filesize_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
    std::unordered_map<std::string, uint64_t> offsets_;
};

// bf16 (top 16 bits of fp32) -> float, host side
static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}
