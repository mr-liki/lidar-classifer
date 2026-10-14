// Included by main.cpp and the focused audit harness; uses its check/skip helpers.
static void section_surface_barycentric() {
    std::printf("\n=== W. FACET / BARYCENTRIC SURFACE ===\n");
    // Full-detail input conservation, including cases a 2D surface cannot mesh.
    {
        std::vector<SurfaceTile> all(1);
        auto& t=all[0];t.cmin[0]=t.cmin[1]=0;t.cmax[0]=t.cmax[1]=1000;
        for(int y=0;y<100;++y) for(int x=0;x<100;++x)
            t.pts.push_back({double(x),double(y),0,2,200,100,50,128});
        for(int i=0;i<10;++i) t.pts.push_back({50,50,double(i),2,200,100,50,128});
        t.pts.push_back({999,999,0,2,200,100,50,128});
        SurfaceParams full;full.ambient=1;
        for(auto method:{SurfaceMethod::Facet,SurfaceMethod::Barycentric}) {
            full.method=method;
            auto m=build_surface_mesh(all,full,1);
            check("every full-detail record is a mesh vertex or a coverage point",
                m.visible_points==t.pts.size() && m.mesh_referenced_points+m.coverage_positions.size()/3==t.pts.size(),
                std::to_string(m.visible_points)+" input, "+std::to_string(m.mesh_referenced_points)+" meshed, "+
                std::to_string(m.coverage_positions.size()/3)+" coverage");
            bool outlier=false;
            for(std::size_t i=0;i<m.coverage_positions.size();i+=3)
                outlier |= m.coverage_positions[i]==999 && m.coverage_positions[i+1]==999;
            check("long-edge filtering retains isolated input as a coverage point",outlier&&m.dropped_long_edges>0);
        }
        all[0].pts={{10,10,0,2,200,100,50,128},{20,20,0,2,200,100,50,128}};
        auto two=build_surface_mesh(all,full,1);
        check("fewer than three points are retained",two.faces.empty()&&two.coverage_positions.size()==6);
        // A sparse leaf at the edge of a dense patch must not become a hard
        // rectangular surface boundary merely because it owns <3 records.
        std::vector<SurfaceTile> sparse(2);
        for(auto& s:sparse) {s.cmin[1]=0;s.cmax[1]=2;s.cmin[2]=0;s.cmax[2]=2;}
        sparse[0].cmin[0]=0;sparse[0].cmax[0]=2;
        sparse[1].cmin[0]=2;sparse[1].cmax[0]=4;
        sparse[0].pts={{1,1,0,2,200,100,50,128}};
        sparse[1].pts={{2.2,0.8,0,2,200,100,50,128},
                       {2.2,1.2,0,2,200,100,50,128},
                       {2.8,0.8,0,2,200,100,50,128},
                       {2.8,1.2,0,2,200,100,50,128}};
        auto joined=build_surface_mesh(sparse,full,1);
        bool sparse_joins=false;
        for(size_t f=0;f<joined.faces.size();f+=3)
            for(int k=0;k<3;++k)
                sparse_joins |= joined.positions[3*joined.faces[f+k]]==1.0;
        check("single-record leaf joins its neighbour's surface",sparse_joins &&
              joined.mesh_referenced_points+joined.coverage_positions.size()/3==5);
        std::vector<SurfaceTile> grid(32*32);
        for(int y=0;y<32;++y) for(int x=0;x<32;++x) {
            auto& cell=grid[y*32+x];
            cell.cmin[0]=x*4;cell.cmax[0]=(x+1)*4;
            cell.cmin[1]=y*4;cell.cmax[1]=(y+1)*4;
            for(int py=0;py<2;++py) for(int px=0;px<2;++px)
                cell.pts.push_back({x*4+1.0+px*2,y*4+1.0+py*2,0,2,200,100,50,128});
        }
        auto indexed=build_surface_mesh(grid,full,3);
        check("indexed neighbour search retains every record across 1024 leaves",
              indexed.visible_points==4096 &&
              indexed.mesh_referenced_points+indexed.coverage_positions.size()/3==4096);
        all[0].pts.clear();
        for(int i=0;i<25;++i) all[0].pts.push_back({double(i*3+10),50,0,2,200,100,50,128});
        auto collinear=build_surface_mesh(all,full,1);
        check("collinear points are all retained",collinear.faces.empty()&&collinear.coverage_positions.size()==75);
        VulkanEngine eng;std::string err;
        if(eng.load(VulkanEngine::find_dll(),err)&&eng.open(640,480,true,false,err)) {
            eng.set_clear_color(0,0,0,1);eng.set_camera_lookat(50,49,200,50,50,0,45,1,1000);
            check("Shading enters surface-only presentation before first mesh upload",
                eng.show_surface(true));
            check("a triangle-free cut uploads and activates native point coverage",
                eng.upload_surface(collinear,1)&&eng.show_surface(true),eng.last_error());
            for(int i=0;i<3;++i) eng.render();
            EngineFrame frame;
            check("all 25 separated collinear records render through native GPU coverage",
                eng.capture_frame(frame)&&frame.count_not_equal(0,0,0)==25);
        } else check("coverage renderer opens",false,err);
    }
    std::vector<SurfaceTile> tiles(2);
    for (int j=0;j<2;++j) {
        auto& t=tiles[j];
        t.cmin[0]=j*50; t.cmax[0]=(j+1)*50;
        t.cmin[1]=0; t.cmax[1]=100; t.cmin[2]=0; t.cmax[2]=10;
        for (int y=0;y<=100;++y) for (int x=j*50;x<(j+1)*50;++x)
            t.pts.push_back({double(x)+0.2,double(y)+0.15,0.0,
                uint8_t(x<50?2:5),10,100,200,uint8_t(x*2)});
    }
    SurfaceParams p; p.ambient=1; p.elevation_deg=90;
    for (int k=0;k<3;++k) { p.class_rgb[2][k]=k==0?255:0; p.class_rgb[5][k]=k==2?255:0; }
    auto facet=build_surface_mesh(tiles,p,1);
    facet.blocks={uint32_t(facet.faces.size())};
    p.method=SurfaceMethod::Barycentric;
    auto vc=build_surface_mesh(tiles,p,1); vc.blocks=facet.blocks;
    check("barycentric retains every facet face and fallback colour byte",
        vc.faces==facet.faces && vc.positions==facet.positions && vc.colors==facet.colors);
    check("one RGB per remapped vertex; facet has no vertex stream",
        vc.vertex_colors.size()==vc.positions.size() && facet.vertex_colors.empty());
    bool own=true;
    for (std::size_t i=0;i<vc.vertices();++i) {
        const bool red=vc.positions[3*i]<50;
        own &= vc.vertex_colors[3*i]==(red?255:0) && vc.vertex_colors[3*i+1]==0
            && vc.vertex_colors[3*i+2]==(red?0:255);
    }
    check("each vertex uses its own class, including border halo vertices",own);
    p.class_visible[5]=0;
    auto hidden=build_surface_mesh(tiles,p,1);
    bool no_hidden=!hidden.faces.empty();
    for (std::size_t i=0;i<hidden.vertices();++i) no_hidden &= hidden.positions[3*i]<50;
    check("hidden classes contribute neither vertices nor triangles",no_hidden);
    p.class_visible[5]=1; p.color_source=SurfaceColor::Elevation; p.z_min=0;p.z_max=100;
    auto slope=tiles;
    for (auto& t:slope) for(auto& q:t.pts) q.z=q.x;
    auto elev=build_surface_mesh(slope,p,1);
    bool ramp=true;
    for (std::size_t i=0;i<elev.vertices();++i) {
        float rgb[3]; evaluate_ramp(p.ramp,float(elev.positions[3*i+2]/100),rgb);
        for(int k=0;k<3;++k) ramp &= elev.vertex_colors[3*i+k]==uint8_t(rgb[k]*255+0.5);
    }
    check("elevation ramp uses each vertex's value",ramp);
    // Independently known plane normal checks area weighting and halo accumulation.
    p.color_source=SurfaceColor::Class;p.ambient=0.1;p.elevation_deg=40;
    auto lit=build_surface_mesh(slope,p,1); double L[3]; surface_light_vector(p.azimuth_deg,p.elevation_deg,L);
    const double shade=std::max(0.1,(-L[0]+L[2])/std::sqrt(2.0));
    bool normals=true;
    for(std::size_t i=0;i<lit.vertices();++i) {
        const int k=lit.positions[3*i]<50?0:2;
        normals &= lit.vertex_colors[3*i+k]==uint8_t(255*shade+0.5);
    }
    check("area weighted plane normals match analytic shade across tile borders",normals);
    auto benchmark=make_terrain_tiles(150000,31);
    p=SurfaceParams();
    double ft=0,bt=0;
    for(int i=0;i<3;++i) {
        p.method=SurfaceMethod::Facet; ft+=build_surface_mesh(benchmark,p,3).build_ms;
        p.method=SurfaceMethod::Barycentric; bt+=build_surface_mesh(benchmark,p,3).build_ms;
    }
    std::printf("[W timing] 150k points / 18 tiles: facet %.1f ms, barycentric %.1f ms, ratio %.3f (3-run means)\n",ft/3,bt/3,bt/ft);

    // Measure build + indexed upload on exactly the same cut in one renderer.
    {
        VulkanEngine eng; std::string err;
        if (eng.load(VulkanEngine::find_dll(),err) && eng.open(640,480,true,false,err)) {
            double total[2]={0,0}, upload[2]={0,0}; uint64_t revision=0;
            for(int run=0;run<3;++run) for(int method=0;method<2;++method) {
                p.method=SurfaceMethod(method);
                auto mesh=build_surface_mesh(benchmark,p,3);
                mesh.blocks={uint32_t(mesh.faces.size())};
                const auto begin=std::chrono::steady_clock::now();
                check("timed indexed upload succeeds",eng.upload_surface(mesh,++revision),eng.last_error());
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                upload[method]+=ms;total[method]+=mesh.build_ms+ms;
                eng.render(); // activate/retire between runs, preserving normal swap budget
            }
            std::printf("[W build+upload] same 150k cut, 3-run means: facet %.1f + %.1f = %.1f ms; barycentric %.1f + %.1f = %.1f ms; ratio %.3f\n",
                (total[0]-upload[0])/3,upload[0]/3,total[0]/3,
                (total[1]-upload[1])/3,upload[1]/3,total[1]/3,total[1]/total[0]);
        } else check("timing renderer opens",false,err);
    }

    const std::string current=VulkanEngine::find_dll();
    auto capture=[&](const std::string& dll,const SurfaceMesh& mesh,EngineFrame& frame,bool expect_vc) {
        VulkanEngine eng; std::string err;
        if(!eng.load(dll,err)||!eng.open(640,480,true,false,err)) { check("W renderer opens",false,err);return false; }
        check("optional vertex export matches engine version",eng.has_vertex_surface()==expect_vc);
        eng.set_clear_color(0,0,0,1);
        eng.set_camera_lookat(50,49,200,50,50,0,45,1,1000);
        if(!eng.upload_surface(mesh,1)||!eng.show_surface(true)) {check("W upload",false,eng.last_error());return false;}
        if (!expect_vc && !mesh.vertex_colors.empty())
            check("repeated old-engine fallback upload succeeds",eng.upload_surface(mesh,2));
        for(int i=0;i<3;++i) eng.render();
        return eng.capture_frame(frame);
    };
    if(current.empty()) {skip("W GPU", "engine DLL absent");return;}
    EngineFrame a,b,c;
    if(!capture(current,facet,a,true)||!capture(current,vc,b,true)) return;
    uint64_t grad=0,gaps=0; std::unordered_set<uint32_t> distinct;
    // A five pixel band crossing the known x=50 class boundary, central interior.
    for(uint32_t y=b.height/2-100;y<b.height/2+100;++y)
      for(uint32_t x=b.width/2-2;x<b.width/2+3;++x) {
        const auto* q=b.at(x,y);
        grad += q[0]>5 && q[2]>5;
        gaps += q[0]==0 && q[1]==0 && q[2]==0;
        distinct.insert((uint32_t(q[0])<<16)|(uint32_t(q[1])<<8)|q[2]);
    }
    check("GPU boundary five pixel band contains an interpolated gradient",grad>100&&distinct.size()>2,
        std::to_string(grad)+" intermediate pixels, "+std::to_string(distinct.size())+" colours");
    check("GPU tile/class border contains no black gaps",gaps==0,std::to_string(gaps));
    const char* legacy=std::getenv("LNT_LEGACY_VULKAN_DLL");
    if(legacy) {
        if(capture(legacy,facet,c,false))
            check("facet pixels equal pre-feature engine byte for byte",a.rgba==c.rgba);
        EngineFrame fallback;
        auto legacy_vc=vc;
        legacy_vc.coverage_positions.clear();legacy_vc.coverage_colors.clear();
        if(capture(legacy,legacy_vc,fallback,false))
            check("old engine falls back to facet with identical pixels",fallback.rgba==c.rgba);
        auto single=tiles;
        for(auto& t:single) for(auto& q:t.pts) q.cls=2;
        SurfaceParams single_p; single_p.ambient=1;single_p.elevation_deg=90;
        single_p.class_rgb[2][0]=200;single_p.class_rgb[2][1]=100;single_p.class_rgb[2][2]=50;
        auto sm=build_surface_mesh(single,single_p,1);sm.blocks={uint32_t(sm.faces.size())};
        sm.coverage_positions.clear();sm.coverage_colors.clear();
        EngineFrame old_single,new_single;
        if(capture(legacy,sm,old_single,false)&&capture(current,sm,new_single,true))
            check("single class capture before/after has zero differing bytes",old_single.rgba==new_single.rgba);
    } else skip("pre-feature engine parity", "set LNT_LEGACY_VULKAN_DLL to the saved DLL");

    // ---- crisp shaded-classification model (VTK parity) -------------------------------
    // The native Shading mesh is uploaded through the engine's shaded-class path, which lights
    // it with gui/shading_display.py's model. Two things must hold: the mesher must hand that
    // path the per-vertex class stream and the mixed-face split it indexes, and
    // LNTShadingModes.h must reproduce the CPU model exactly, including its default-selection
    // equivalence with the legacy facet shade.
    {
        SurfaceParams K;
        K.class_rgb[2][0]=255; K.class_rgb[2][1]=0; K.class_rgb[2][2]=0;
        K.class_rgb[5][0]=0;   K.class_rgb[5][1]=0; K.class_rgb[5][2]=255;
        K.method = SurfaceMethod::Barycentric;
        const auto km = build_surface_mesh(tiles, K, 1);
        check("every shaded-class mesh vertex carries its own class id",
              km.vertex_class.size() == km.vertices(),
              std::to_string(km.vertex_class.size()) + " class bytes for " +
              std::to_string(km.vertices()) + " vertices");
        std::vector<int32_t> expect;
        for (std::size_t f = 0; f < km.triangles(); ++f) {
            const uint32_t k0 = lnt::detail::surface_class_key(K, km.vertex_class[km.faces[3*f+0]]);
            const uint32_t k1 = lnt::detail::surface_class_key(K, km.vertex_class[km.faces[3*f+1]]);
            const uint32_t k2 = lnt::detail::surface_class_key(K, km.vertex_class[km.faces[3*f+2]]);
            if (k0 != k1 || k0 != k2) expect.push_back(int32_t(f));
        }
        check("mixed-face list is exactly the faces whose three display colours differ",
              km.mixed_faces == expect,
              std::to_string(km.mixed_faces.size()) + " mixed of " +
              std::to_string(km.triangles()) + " faces");
        check("the class boundary really does produce mixed faces (the check is not vacuous)",
              !km.mixed_faces.empty());
        check("a single-class (facet) mesh carries the class stream too",
              facet.vertex_class.size() == facet.vertices());
        VulkanEngine probe; std::string perr;
        if (probe.load(VulkanEngine::find_dll(), perr))
            check("the engine DLL exports the crisp shaded-class surface path",
                  probe.has_shaded_class_surface());
        else skip("crisp shaded-class engine path", perr);
    }
    {
        const double kDeg = 3.14159265358979323846 / 180.0;
        SurfaceShading s;                       // defaults == the VTK view's defaults
        auto legacy = [&](const double n[3]) {
            const double z = (90.0 - s.light_elevation_deg) * kDeg;
            const double a = (360.0 - s.azimuth_deg + 90.0) * kDeg;
            const double L[3] = {std::sin(z)*std::cos(a), std::sin(z)*std::sin(a), std::cos(z)};
            const double ndl = n[0]*L[0] + n[1]*L[1] + n[2]*L[2];
            return std::min(1.0, std::max(ndl, double(s.ambient)));
        };
        const double normals[5][3] = {{0,0,1},{0.7071,0,0.7071},{-0.7071,0,0.7071},
                                      {0,0.7071,0.7071},{0.5,-0.5,0.7071}};
        double worst = 0.0;
        for (const auto& n : normals) {
            const double nn = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            const double u[3] = {n[0]/nn, n[1]/nn, n[2]/nn};
            worst = std::max(worst, std::abs(double(crisp_face_shade(s, u)) - legacy(u)));
        }
        check("at the default Sharpness 45 the crisp shade equals the legacy max(N.L, ambient)",
              worst < 1e-6, "worst " + fmt_e(worst));
        SurfaceShading hot = s; hot.sharpness = 200.0f;
        float legacy_r = 0.0f, overdrive = 0.0f;
        shading_sharpness_response(hot.sharpness, legacy_r, overdrive);
        check("Sharpness 200 reaches the documented overdrive response",
              std::abs(overdrive - 0.90f) < 1e-6f && std::abs(legacy_r - 1.0f) < 1e-6f,
              "legacy " + std::to_string(legacy_r) + " overdrive " + std::to_string(overdrive));
        const float gain = shading_contrast_gain(legacy_r, overdrive);
        const float eff = shading_effective_light_elevation(hot.light_elevation_deg, overdrive);
        check("overdrive lowers the effective light elevation and raises the contrast gain",
              eff < hot.light_elevation_deg && gain > 1.0f,
              "elevation " + std::to_string(eff) + " gain " + std::to_string(gain));
        float key = 0.0f, fill = 0.0f;
        shading_key_fill(hot, key, fill);
        check("the key/fill rig conserves directional energy as Sharpness rises",
              std::abs((key + fill) - hot.directional_energy) < 1e-5f && key > 0.0f,
              "key " + std::to_string(key) + " fill " + std::to_string(fill));
    }
}
