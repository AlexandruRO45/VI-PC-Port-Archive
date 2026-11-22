extern "C" int delayed_value() {
#ifdef ALTER_RETURN
    return 47;
#else
    return 31;
#endif
}

extern "C" int delayed_ordinal() {
#ifdef ALTER_RETURN
    return 61;
#else
    return 59;
#endif
}
