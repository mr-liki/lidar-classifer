// tool_link_check.cpp - links and runs the two non-inline helpers of the split
// build: describe_file() (LNTReader.cpp) and describe_streaming()
// (src/LNTStreamer.cpp).
//
// WHY THIS FILE EXISTS. Both are declared in headers and defined in translation
// units that main.cpp must NOT reference - the single-command harness build
// (`g++ -O3 -std=c++17 main.cpp -o main`) has no second object file to link
// against, so main.cpp calling them would break that build. The consequence is
// that nothing else in the tree exercises them, and a compile-only check cannot
// catch a link error or a crash inside them. This is that exercise:
//
//     g++ -O3 -std=c++17 -Wall -Wextra tool_link_check.cpp LNTReader.cpp
//         src/LNTStreamer.cpp -o tool_link_check && ./tool_link_check test.lnt
// (one command in a shell; the line break above is only for readability here)
//
// It is a smoke test, not a verdict: it prints whatever the two formatters
// produce for a real file after a real (selection + drain) frame, and exits
// non-zero only if the file cannot be opened, the cache cannot start or the
// streamer cannot attach.

#include <cstdio>
#include <string>

#include "include/LNTReader.h"
#include "include/LNTStreamer.h"

using namespace lnt;

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "test.lnt";

    Reader rd;
    if (!rd.open(path)) {
        std::printf("open failed: %s\n", rd.error().c_str());
        return 1;
    }

    LntCacheManager cache(8ull * 1024 * 1024);
    std::string err;
    if (!cache.open(path, err)) {
        std::printf("cache open failed: %s\n", err.c_str());
        return 1;
    }

    LntStreamer streamer;
    if (!streamer.attach(rd, &cache, err)) {
        std::printf("attach failed: %s\n", err.c_str());
        return 1;
    }

    // A camera outside the envelope looking at its centre, so the printout has a
    // non-empty cut, a non-zero cull count and real residency to report.
    const BoundingBox env = rd.get_bounds();
    LNTCamera cam;
    cam.eye[0] = env.mid(0) - env.extent(0) * 2.0;
    cam.eye[1] = env.mid(1) - env.extent(1) * 2.0;
    cam.eye[2] = env.mid(2) + env.extent(2) * 1.5;
    for (int i = 0; i < 3; ++i) cam.target[i] = env.mid(i);
    for (int i = 0; i < 3; ++i) cam.origin[i] = rd.origin(i);
    cam.rebuild();

    streamer.update_visible_nodes(cam);
    if (!cache.drain(20000)) std::printf("WARNING: the loader did not drain\n");
    streamer.update_visible_nodes(cam);   // now with the payloads resident

    std::printf("%s", describe_file(rd).c_str());
    std::printf("%s", describe_streaming(streamer, cache).c_str());

    cache.close();
    return 0;
}
