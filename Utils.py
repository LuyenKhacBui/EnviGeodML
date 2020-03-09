def prgrsTime(disstr, prgrs_prct, prgrs_runtime):
	prgrs_esttime = prgrs_runtime / prgrs_prct * (100 - prgrs_prct)
#	print disstr.expandtabs(8) + " ---> " + str(int(round(prgrs_runtime))) + " s (", str(round(float(prgrs_runtime)/60, 1)) + " mins) / " + str(int(round(prgrs_esttime))) + " s (" + str(round(float(prgrs_esttime/60), 1)) + " mins)"
	print (disstr.expandtabs(8) + " ---> " + str(int(round(prgrs_runtime))) + " s (", str(round(float(prgrs_runtime)/60, 1)) + " mins) / " + str(int(round(prgrs_esttime))) + " s (" + str(round(float(prgrs_esttime/60), 1)) + " mins)")