#define main lnt_full_harness_main
#include "../native/lnt/main.cpp"
#undef main
int main() {
    section_surface("native/lnt/test_shading.lnt");
    section_surface_barycentric();
    std::printf("%d/%d checks passed; %d skipped\n",g_checks-g_failed,g_checks,g_skipped);
    return g_failed?1:0;
}
