	; Fetch register 0 and an ordinary register 1. Export all fetched words verbatim.
	s_swappc_b64 s[4:5], s[4:5]
	exp pos0 v8, v9, v10, v11 done
	exp param0 v4, v5, v6, v7
	s_endpgm
