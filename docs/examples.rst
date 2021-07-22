Examples
========

py_peak_splitting.th_filter
+++++++++++++++++++++++++++
.. code-block:: python

    """This example demonstrates the use of py_peak_splitting.th_filter.Th_filter for the purpose of removing RTN a-like 
	noise casued by peak splitting artefacts. 

    See `here <https://github.com/OWI-Lab/py_peak_splitting/tree/main/demo>` for jupyter notbooks illustrating the use 
	on corrupted measurement data obtained from an offshore wind turbine, including a offset/trend calibration of the 
	reconstruction.
	
	Notes on this example: 
	A. this example does not demonstrate the offset/trend calibration of the 
	reconstructed signal for the purpose of 1) illustrating the insensitivty of the reconstruction w.r.t. the mean of the
	raw signal, be able to visually comapre the noise free and reconstructed signal.
	
    """

	from py_peak_splitting.th_filter import ThFilter as Th
	from py_peak_splitting import simulate_ps_signals as sim
	import matplotlib.pyplot as plt	
	
	# create an artificial signal containing a additive RTN noise source
	Fs = 100
	duration = 20 
	t = np.linspace(0, duration-1/Fs, Fs*duration)
	arr0 = sim.dummy_signal(f=[0.2, 0.456, 0.32], Fs=Fs, duration=duration, amplitude=None, phi=None)
	arr = arr0 + sim._ps2(arr=arr0, jump_v=[1, 1], jump_s=[0.05,0.1], p=[0.75, 0.75], scale=1, jot=[2,4], scale_by_amplitude=False)
	
	# Perform a reconstruction
	th = Th(arr=arr)  
	thv = th.estimate_th(aggressive=True, th=500, plot=False)
	rec, clu, lab = th.recon_th(th=thv, method='polyfit', nbuffer=25)
	rec = rec.flatten() + arr0[0]

	# visualise the simulated signals
	fig, ax = plt.subplots(1,1,figsize=(15,5))
	ax.plot(t, arr0, label='noise free')
	ax.plot(t, arr, label='corrupted')
	ax.plot(t, rec, label='recostructed')
	plt.show()


py_peak_splitting.hampel_filter
+++++++++++++++++++++++++++++++
.. code-block:: python

    """This example demonstrates the use of py_peak_splitting.hampel_filter.Hampel for the purpose of removing RTN a-like 
	noise casued by peak splitting artefacts. 

    See `here <https://github.com/OWI-Lab/py_peak_splitting/tree/main/demo>` for jupyter notbooks illustrating the use 
	on corrupted measurement data obtained from an offshore wind turbine, including a offset/trend calibration of the 
	reconstruction.
		
	Notes on this example: 
	A. this example does not demonstrate the offset/trend calibration of the 
	reconstructed signal for the purpose of 1) illustrating the insensitivty of the reconstruction w.r.t. the mean of the
	raw signal, be able to visually comapre the noise free and reconstructed signal.	
    """

	from py_peak_splitting.hampel_filter import Hampel as Ha
	from py_peak_splitting import simulate_ps_signals as sim
	import matplotlib.pyplot as plt	

	# create an artificial signal containing a additive RTN noise source
	Fs = 100
	duration = 20 
	t = np.linspace(0, duration-1/Fs, Fs*duration)
	arr0 = sim.dummy_signal(f=[0.2, 0.456, 0.32], Fs=Fs, duration=duration, amplitude=None, phi=None)
	arr = arr0 + sim._ps2(arr=arr0, jump_v=[1, 1], jump_s=[0.05,0.1], p=[0.75, 0.75], scale=1, jot=[2,4], scale_by_amplitude=False)

	# Perform a reconstruction
	h = Ha(arr=arr, th='sig', th_val= 3, re='plf', nperseg=100)
	rec = h.reconstruction         

	# visualise the simulated signals
	fig, ax = plt.subplots(1,1,figsize=(15,5))
	ax.plot(t, arr0, lw=2, c='b', label='noise free')
	ax.plot(t, arr, c ='r', label='corrupted')
	ax.plot(t, rec, lw=2, c='g', label='recostructed')
	ax.legend()
	plt.show()

py_peak_splitting.simulate_ps_signals
+++++++++++++++++++++++++++++++++++++
.. code-block:: python

    """This example demonstrates the use of py_peak_splitting.th_filter.Th_filter for the purpose of removing RTN a-like 
	noise casued by peak splitting artefacts. Note, this example does not demonstrate the offset/trend calibration of the 
	reconstructed signal for the purpose of 1) illustrating the insensitivty of the reconstruction w.r.t. the mean of the
	raw signal, be able to visually comapre the noise free and reconstructed signal.

    """

	from py_peak_splitting import simulate_ps_signals as sim
	import matplotlib.pyplot as plt	
		
	# create an artificial signal containing a additive RTN noise source
	Fs = 100
	duration = 5 
	t = np.linspace(0, duration-1/Fs, Fs*duration)
	arr0 = sim.dummy_signal(f=[0.2, 0.456, 0.32], Fs=Fs, duration=duration, amplitude=None, phi=None)
	arr1 = arr0 + sim.ps1(arr=arr0, jump_v=[1, 1], jump_s=[0, 0], jot=[2, 4], scale=1, scale_by_amplitude=False)
	arr2 = arr0 + sim.ps2(arr=arr0, jump_v=[1, 1], jump_s=[0, 0], p=[0.75, 0.75], scale=1, jot=[2,4], scale_by_amplitude=False)
	arr3 = arr0 + sim.ps3(arr=arr0,  jump_v=[1], jump_s=[0], scale=1, jot=[2], scale_by_amplitude=True)

	# visualise the simulated signals
	fig, ax = plt.subplots(1,1,figsize=(15,5))
	ax.plot(t, arr0, label='noise free')
	ax.plot(t, arr1, label='corrupted_1')
	fig, ax = plt.subplots(1,1,figsize=(15,5))
	ax.plot(t, arr0, label='noise free')
	ax.plot(t, arr2, label='corrupted_2')
	fig, ax = plt.subplots(1,1,figsize=(15,5))
	ax.plot(t, arr0, label='noise free')
	ax.plot(t, arr3, label='corrupted_3')
	plt.show()