// Savage Wind: force-included (-include) into the FidelityFX SDK's C++ sources, so that they build with clang/gcc
// (MinGW-w64 and Linux) unchanged: what MSVC's headers bring in on their own, and its _s functions where the C library
// has none.
#pragma once
#ifdef __cplusplus
#include <cmath>
#include <codecvt>  // (std::wstring_convert, which ffx_vk.cpp uses for the debug names: MSVC's headers bring it)
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <locale>
#include <new>
#include <string>
#endif

#if !defined(_MSC_VER)
#define swprintf_s swprintf  // (the SDK passes the size)
#if defined(__cplusplus) && !defined(_countof)
template <typename T, size_t N> constexpr size_t ffx_countof_(T (&)[N]) { return N; }
#define _countof(a) ffx_countof_(a)  // (stdlib.h's on Windows; glibc has none)
#endif
#if !defined(_WIN32)
#include <stddef.h>
#include <wchar.h>
static inline int wcscpy_s(wchar_t *d, size_t n, const wchar_t *s) {
    size_t l = wcslen(s);
    if (l >= n) l = n - 1;
    wmemcpy(d, s, l);
    d[l] = 0;
    return 0;
}
static inline int wcscat_s(wchar_t *d, size_t n, const wchar_t *s) {
    size_t l = wcslen(d);
    return l < n ? wcscpy_s(d + l, n - l, s) : 0;
}
static inline int strcpy_s(char *d, size_t n, const char *s) {
    size_t l = strlen(s);
    if (l >= n) l = n - 1;
    memcpy(d, s, l);
    d[l] = 0;
    return 0;
}
#ifdef __cplusplus
template <size_t N> static inline int wcscpy_s(wchar_t (&d)[N], const wchar_t *s) { return wcscpy_s(d, N, s); }
template <size_t N> static inline int strcpy_s(char (&d)[N], const char *s) { return strcpy_s(d, N, s); }
template <size_t N, typename... A> static inline int sprintf_s(char (&d)[N], const char *f, A... a) { return snprintf(d, N, f, a...); }
template <typename... A> static inline int sprintf_s(char *d, size_t n, const char *f, A... a) { return snprintf(d, n, f, a...); }
#endif
#ifndef FFX_UNUSED
#define FFX_UNUSED(x) ((void)(x))  // (ffx_message.cpp: defined by the SDK's Windows headers only)
#endif
#endif
#endif
