"""Runtime GDK : état HDR de l'écran (XDisplay) et connectivité réseau (XNetworking) du PC.

Structures et signatures : XDisplay.h et XNetworking.h du GDK public (version 261000).
Les valeurs viennent de Windows (DXGI, Network List Manager) : rien n'est inventé.
"""
import ctypes

from test_runtime_cpp import S_OK, GdkInterface
from test_runtime_events import E_INVALIDARG, E_NOTIMPL, EVENT_CALLBACK

HDR = ("03F0FE74-FDD9-4E5C-B630-F9339C47ACC5", "35F07670-706E-4BFB-9476-090798C5EBF3")
NETWORKING = ("37E56907-2F10-41E8-B72F-36EDB185331A", "BF2346B2-39AF-4658-B5EA-44713C7E83B3")
HDR_UNKNOWN, HDR_ENABLED, HDR_DISABLED = 0, 1, 2
PREFER_HDR = 0
LEVEL_NONE, LEVEL_CONSTRAINED = 1, 4


class HdrModeInfo(ctypes.Structure):
    _fields_ = [("min_luminance", ctypes.c_float), ("max_luminance", ctypes.c_float),
                ("max_full_frame_luminance", ctypes.c_float)]


class ConnectivityHint(ctypes.Structure):
    _fields_ = [("level", ctypes.c_uint32), ("cost", ctypes.c_uint32), ("iana_interface_type", ctypes.c_uint32),
                ("network_initialized", ctypes.c_bool), ("approaching_data_limit", ctypes.c_bool),
                ("over_data_limit", ctypes.c_bool), ("roaming", ctypes.c_bool)]


def test_hdr_mode_reports_the_pc_display(runtime_dir):
    try_enable = GdkInterface(runtime_dir, *HDR).method(4, ctypes.c_uint32, ctypes.c_uint32,
                                                        ctypes.POINTER(HdrModeInfo))
    info = HdrModeInfo()
    result = try_enable(PREFER_HDR, ctypes.byref(info))
    assert result in (HDR_UNKNOWN, HDR_ENABLED, HDR_DISABLED)
    if result == HDR_ENABLED:
        assert info.max_luminance > info.min_luminance >= 0
    assert try_enable(PREFER_HDR, None) == result                   # infos facultatives


def test_connectivity_hint_comes_from_windows(runtime_dir):
    get = GdkInterface(runtime_dir, *NETWORKING).method(15, ctypes.c_uint32, ctypes.POINTER(ConnectivityHint))
    hint = ConnectivityHint()
    assert get(ctypes.byref(hint)) == S_OK
    assert LEVEL_NONE <= hint.level <= LEVEL_CONSTRAINED and hint.network_initialized
    assert get(None) == E_INVALIDARG


def test_connectivity_changes_can_be_subscribed(runtime_dir):
    api = GdkInterface(runtime_dir, *NETWORKING)
    subscribe = api.method(16, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_uint64))
    unsubscribe = api.method(17, ctypes.c_bool, ctypes.c_uint64, ctypes.c_bool)
    callback, token = EVENT_CALLBACK(lambda context, hint: None), ctypes.c_uint64()
    assert subscribe(None, None, ctypes.cast(callback, ctypes.c_void_p), ctypes.byref(token)) == S_OK
    assert unsubscribe(token.value, True) is True and unsubscribe(token.value, True) is False


def test_other_networking_methods_are_not_implemented(runtime_dir):
    port = GdkInterface(runtime_dir, *NETWORKING).method(3, ctypes.c_uint32, ctypes.c_void_p)
    assert port(None) == E_NOTIMPL                     # XNetworkingQueryPreferredLocalUdpMultiplayerPort
