// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2026 RealSense, Inc. All Rights Reserved.

// Drives the full viewer through 20 depth-only Start/Stop cycles and samples
// the process's "private bytes" after each cycle. Exercises the real viewer
// code path (ux-window loop, gc_streams, post-processing filters, syncer, etc.).
//
// Windows-only: the leak detector reads `PROCESS_MEMORY_COUNTERS_EX.PrivateUsage`
// (= Task Manager's "Private Bytes" / "private commit"). `getrusage().ru_maxrss`
// on Linux/macOS is the *peak* RSS — monotonically non-decreasing — and is
// therefore useless for measuring per-cycle leak rate. A proper Linux metric
// (/proc/self/status VmRSS) and macOS metric (task_info phys_footprint) is
// follow-up work; until then the test only registers on Windows.
//
// Results are written to viewer_mem_leak_results.csv next to the executable.
// Plot with plot_mem.py from the scratchpad.
//
// Run headlessly:
//   realsense-viewer-tests --auto -r mem_leak_depth_start_stop

#ifdef _WIN32

#include "viewer-test-helpers.h"

#include "post-processing-filters.h"
#include "viewer.h"

#include <librealsense2/rs.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <windows.h>
#include <psapi.h>
#pragma comment( lib, "psapi.lib" )


namespace {

size_t get_private_bytes()
{
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if( !GetProcessMemoryInfo( GetCurrentProcess(),
                               reinterpret_cast< PROCESS_MEMORY_COUNTERS * >( &pmc ),
                               sizeof( pmc ) ) )
    {
        std::cerr << "[mem-leak] WARNING: GetProcessMemoryInfo failed, GLE="
                  << GetLastError() << std::endl;
        return 0;  // caller asserts baseline > 0 and propagates the zero into the slope
    }
    return static_cast< size_t >( pmc.PrivateUsage );
}

double to_mb( size_t bytes ) { return bytes / ( 1024.0 * 1024.0 ); }

// Returns the subdevice that exposes a DEPTH stream (the Stereo Module on D4xx).
std::shared_ptr< rs2::subdevice_model > find_depth_subdevice( rs2::device_model & dm )
{
    for( auto && sub : dm.subdevices )
        for( auto && p : sub->profiles )
            if( p.stream_type() == RS2_STREAM_DEPTH )
                return sub;
    return nullptr;
}

// Narrow the depth subdevice's selection to depth-only (turn IR streams off).
// stream_enabled is keyed by stream unique-id; flipping it directly is what the
// viewer's own checkboxes do (subdevice-model.cpp:782, 1016).
void select_depth_only( std::shared_ptr< rs2::subdevice_model > sub )
{
    for( auto && p : sub->profiles )
        sub->stream_enabled[ p.unique_id() ] = ( p.stream_type() == RS2_STREAM_DEPTH );
}

}  // namespace


VIEWER_TEST( "streaming", "mem_leak_depth_start_stop" )
{
    constexpr int   ITERATIONS      = 20;
    constexpr float STREAM_DURATION = 10.0f;  // longer cycles amplify any per-cycle leak
    constexpr float IDLE_DURATION   = 5.0f;

    // Leak verdict (linear-regression slope on iters after warmup).
    // First few iterations include one-time allocations and steady-state pool
    // ramp-up that are not part of the per-cycle leak — skip them.
    constexpr int   WARMUP_ITERS               = 3;
    constexpr float LEAK_THRESHOLD_MB_PER_ITER = 1.0f;

    auto & model = test.find_first_device_or_exit();
    auto depth = find_depth_subdevice( model );
    IM_CHECK( depth != nullptr );
    if( ! depth )
        return;

    select_depth_only( depth );

    const std::string out_path = "viewer_mem_leak_results.csv";
    std::ofstream     out( out_path );
    IM_CHECK( out.is_open() );  // a silently-failing ofstream would no-op every write below

    out << "iteration,private_bytes_mb" << std::endl;

    const auto baseline = get_private_bytes();
    IM_CHECK( baseline > 0 );  // 0 means GetProcessMemoryInfo failed; would skew the slope
    std::cout << "[mem-leak] Device: " << model.dev.get_info( RS2_CAMERA_INFO_NAME )
              << "  SN " << model.dev.get_info( RS2_CAMERA_INFO_SERIAL_NUMBER ) << std::endl;
    std::cout << "[mem-leak] Sensor: " << depth->s->get_info( RS2_CAMERA_INFO_NAME ) << std::endl;
    std::cout << "[mem-leak] Baseline: " << std::fixed << std::setprecision( 2 )
              << to_mb( baseline ) << " MB" << std::endl;
    out << 0 << "," << std::fixed << std::setprecision( 2 ) << to_mb( baseline ) << std::endl;

    std::vector< double > samples_mb;
    samples_mb.reserve( ITERATIONS );

    for( int i = 1; i <= ITERATIONS; ++i )
    {
        test.click_stream_toggle_on( model, depth );
        test.sleep( STREAM_DURATION );
        test.click_stream_toggle_off( model, depth );
        test.sleep( IDLE_DURATION );

        const auto used = get_private_bytes();
        const auto mb   = to_mb( used );
        samples_mb.push_back( mb );

        // Snapshot viewer state to find accumulating containers.
        size_t streams_size = 0, streams_origin_size = 0, ppf_frames_queue_size = 0;
        {
            std::lock_guard< std::mutex > lock( test.viewer_model.streams_mutex );
            streams_size          = test.viewer_model.streams.size();
            streams_origin_size   = test.viewer_model.streams_origin.size();
            ppf_frames_queue_size = test.viewer_model.ppf.frames_queue.size();
        }

        const double delta = mb - to_mb( baseline );
        std::cout << "[mem-leak] iter " << std::setw( 2 ) << i << " : " << std::fixed
                  << std::setprecision( 2 ) << mb
                  << " MB  (delta vs baseline: " << std::showpos << delta << std::noshowpos
                  << " MB)"
                  << "  | streams=" << streams_size
                  << "  streams_origin=" << streams_origin_size
                  << "  ppf.frames_queue=" << ppf_frames_queue_size
                  << std::endl;
        out << i << "," << std::fixed << std::setprecision( 2 ) << mb << std::endl;
    }

    std::cout << "[mem-leak] Results written to " << out_path << std::endl;
    IM_CHECK( ! model.is_streaming() );

    // ---- Leak verdict --------------------------------------------------
    // Ordinary least-squares slope (MB per iteration) on iters [WARMUP+1 .. ITERATIONS].
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int    n = 0;
    for( int i = WARMUP_ITERS; i < ITERATIONS; ++i )  // samples_mb[0] is iter 1
    {
        double x = static_cast< double >( i + 1 );  // 1-based iter index
        double y = samples_mb[ i ];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
        ++n;
    }
    // Guard the OLS denominator. With ITERATIONS=20, WARMUP_ITERS=3 we get n=17 here,
    // but a future tweak to these constants could degenerate the fit — fail loudly
    // rather than producing NaN (which would silently fail the slope IM_CHECK below).
    IM_CHECK( n >= 2 );
    const double denom = n * sum_xx - sum_x * sum_x;
    IM_CHECK( denom > 0.0 );  // distinct integer x's → guaranteed > 0 when n >= 2, but make it explicit
    const double slope = ( n * sum_xy - sum_x * sum_y ) / denom;

    std::cout << "[mem-leak] linear-fit slope (iters " << ( WARMUP_ITERS + 1 ) << ".."
              << ITERATIONS << "): " << std::fixed << std::setprecision( 2 ) << slope
              << " MB/iter  (threshold: " << LEAK_THRESHOLD_MB_PER_ITER << " MB/iter)"
              << std::endl;
    if( slope > LEAK_THRESHOLD_MB_PER_ITER )
        std::cout << "[mem-leak] VERDICT: LEAK DETECTED" << std::endl;
    else
        std::cout << "[mem-leak] VERDICT: no leak" << std::endl;

    IM_CHECK( slope <= LEAK_THRESHOLD_MB_PER_ITER );
}

#endif  // _WIN32
