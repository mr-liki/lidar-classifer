#include "../native/lnt/include/LNTCameraController.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    lnt::LntCameraController ctl;
    ctl.set_target(680500., 5000500., 100.);
    ctl.set_lens(45., 1., 40000.);
    ctl.set_distance(1e-4);
    lnt::LNTCamera camera;
    double origin[3] = {680000., 5000000., 0.};
    assert(ctl.apply_to(camera, origin, 4. / 3., 480));
    assert(camera.ortho_half_height < 1e-4 && camera.near_clip == 1.);
    std::cout << "PASS close orthographic zoom preserves its scene-depth clipping\n";
    ctl.set_view_preset(0);
    assert(ctl.apply_to(camera, origin, 4. / 3., 480));
    assert(camera.near_clip > 0. && camera.near_clip < ctl.distance());
    std::cout << "PASS perspective near plane permits close-up targets\n";
    ctl.set_distance(100.);
    ctl.zoom(-1000.f);
    for (int i = 0; i < 10; ++i) ctl.update(1.f);
    assert(ctl.distance() == ctl.kMinDistance);
    ctl.set_distance(1e30);
    assert(ctl.distance() == ctl.kMaxDistance);
    std::cout << "PASS smooth zoom converges and both distance limits stay finite\n";
}
