#pragma once
// Suppress CRT assertion/error popup dialogs on Windows.
// Redirects runtime check failures and CRT errors to stderr.
//
// Provides:
//   suppress_popups()          — call at start of main()
//   print_backtrace(ep)        — print stack trace from EXCEPTION_POINTERS
//   run_with_seh(fn, on_crash) — run fn() with SEH; calls on_crash(code) on exception
//
// Link with dbghelp.lib (automatic via #pragma comment).

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <crtdbg.h>
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstdlib>

#pragma comment(lib, "dbghelp.lib")

// Print a stack trace from an exception context.
// Call from an SEH filter (where EXCEPTION_POINTERS* is valid).
inline void print_backtrace(EXCEPTION_POINTERS* ep)
{
    if (!ep || !ep->ContextRecord) return;

    HANDLE proc = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    SymInitialize(proc, NULL, TRUE);

    STACKFRAME64 frame = {};
    CONTEXT ctx = *ep->ContextRecord;
#if defined(_M_X64) || defined(__x86_64__)
    frame.AddrPC.Offset    = ctx.Rip;
    frame.AddrFrame.Offset = ctx.Rbp;
    frame.AddrStack.Offset = ctx.Rsp;
    const DWORD machine_type = IMAGE_FILE_MACHINE_AMD64;
#elif defined(_M_ARM64) || defined(__aarch64__)
    frame.AddrPC.Offset    = ctx.Pc;
    frame.AddrFrame.Offset = ctx.Fp;
    frame.AddrStack.Offset = ctx.Sp;
    const DWORD machine_type = IMAGE_FILE_MACHINE_ARM64;
#else
    fprintf(stderr, "backtrace: (unsupported on this arch)\n");
    SymCleanup(proc);
    return;
#endif
    frame.AddrPC.Mode    = AddrModeFlat;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Mode = AddrModeFlat;

    fprintf(stderr, "backtrace:\n");
    for (int i = 0; i < 32; ++i) {
        if (!StackWalk64(machine_type, proc, thread,
                         &frame, &ctx, NULL,
                         SymFunctionTableAccess64, SymGetModuleBase64, NULL))
            break;
        if (frame.AddrPC.Offset == 0) break;

        char buf[sizeof(SYMBOL_INFO) + 256];
        SYMBOL_INFO* sym = (SYMBOL_INFO*)buf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 256;

        DWORD64 disp = 0;
        if (SymFromAddr(proc, frame.AddrPC.Offset, &disp, sym))
            fprintf(stderr, "  [%2d] %s + 0x%llx\n", i, sym->Name, (unsigned long long)disp);
        else
            fprintf(stderr, "  [%2d] 0x%016llx\n", i, (unsigned long long)frame.AddrPC.Offset);
    }
    fflush(stderr);
    SymCleanup(proc);
}

// Run a callable with SEH protection. Returns true if no exception.
// On crash: prints backtrace + exception code, calls on_crash(code), returns false.
//
// Usage:
//   bool ok = run_with_seh(
//       [&]{ do_work(); },
//       [&](DWORD code){ fprintf(stderr, "CRASH: 0x%08lX\n", code); }
//   );
//
// NOTE: fn and on_crash must not throw C++ exceptions (catch inside if needed).
//       The SEH layer is in a separate plain-C function to satisfy MSVC C2712.

namespace no_popup_detail {

// Store callable pointers in globals to avoid C++ objects inside __try.
inline void (*g_seh_fn)();
inline void (*g_seh_on_crash)(DWORD code);
inline EXCEPTION_POINTERS* g_seh_ep;

inline int seh_filter(EXCEPTION_POINTERS* ep)
{
    g_seh_ep = ep;
    print_backtrace(ep);
    return EXCEPTION_EXECUTE_HANDLER;
}

#pragma warning(push)
#pragma warning(disable: 4611)
inline DWORD seh_call()
{
    __try { g_seh_fn(); } __except (seh_filter(GetExceptionInformation())) {
        return GetExceptionCode();
    }
    return 0;
}
#pragma warning(pop)

} // namespace no_popup_detail

template<typename Fn, typename OnCrash>
inline bool run_with_seh(Fn&& fn, OnCrash&& on_crash)
{
    // Wrap the callable in a plain function pointer via a static lambda.
    // The lambda captures nothing — we pass context through globals.
    static Fn* s_fn;
    static OnCrash* s_on_crash;
    s_fn = &fn;
    s_on_crash = &on_crash;

    no_popup_detail::g_seh_fn = []{ (*s_fn)(); };
    no_popup_detail::g_seh_on_crash = [](DWORD code){ (*s_on_crash)(code); };

    DWORD code = no_popup_detail::seh_call();
    if (code) {
        on_crash(code);
        return false;
    }
    return true;
}

inline void suppress_popups()
{
    // Redirect CRT assert/error/warn to stderr instead of popup dialogs.
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
    // Suppress abort() message box and Windows error reporting.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _set_error_mode(_OUT_TO_STDERR);
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    // Suppress unhandled exception dialog; print backtrace and exit non-zero.
    SetUnhandledExceptionFilter([](EXCEPTION_POINTERS* ep) -> LONG {
        DWORD code = ep ? ep->ExceptionRecord->ExceptionCode : 0;
        fprintf(stderr, "FATAL: unhandled exception 0x%08lX\n", code);
        print_backtrace(ep);
        TerminateProcess(GetCurrentProcess(), code);
        return EXCEPTION_EXECUTE_HANDLER; // unreachable
    });
}
#else
inline void suppress_popups() {}
inline void print_backtrace(void*) {}

template<typename Fn, typename OnCrash>
inline bool run_with_seh(Fn&& fn, OnCrash&&)
{
    fn();
    return true;
}
#endif
