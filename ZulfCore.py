import tkinter as tk
from tkinter import filedialog
import numpy as np
import spectrum as sp
import scipy as sc
import nmrglue as ng
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from dataclasses import dataclass
import types

@dataclass
class Operation:
    """
    A dataclass wrapping the methods with it's parameters and other attributes.
    
    Atributes
    ---------
    outer_instance : object
        Holds reference to the instance of :class:`Spectrum` of which this
        object is an attribute of.
    name :  str
        String reference to the postprocessing operation.
    function : functionType
        Holds function reference/pointer to postprocessing method.
    parameters : dict
        Holds parameters to be used in postprocessing method.
    enabled : bool
        Boolian flag used for enablig/disabling the operation alltogether.
        
    Methods
    -------
    execute(self, x, y, fft)  
        Calls the operation's method while adding settings in
        `~.Operation.paremeters` as kwargs.
    """
    
    outer_instance: object
    name: str
    function: types.FunctionType
    parameters: dict
    enabled: bool
    def execute(self, x, y, fft):
        """
        Parameters
        ----------
        x : numpy.ndarray
            Time signal timestamps.
        y : numpy.ndarray
            Time signal values.
        fft : tuple
            Pair of `numpy.ndarray` containing frequency and spectral values.

        Notes
        -----
        Method returns function call of given postprocessing method it wraps 
        around. Those methods typically return `x`, `y`, `fft` arrays changed
        by postprocessing.
        """
        
        if not self.enabled:
            print(f"Operation {self.name} is disabled")
            return
        return self.function(x, y, fft, **self.parameters)


class ZulfCore:
    """ 
    A class containing main postprocessing and helper methods.
    
    Main postprocessing methods are wrapped in :class:`~.Operation`
    which holds attributes related to given method.
    
    Note
    ----
    Postprocessing method's (such as `open`, `cut`, `debase`, `filter`, etc.)
    signatures always have parameters of `x`, `y`, `fft`. Despite some
    functions only using some, or none of those parameters, it is helpful to 
    have signature compatibility to avoid handling of multiple cases when those
    methods are iteratively called in :method:`Spectrum.__postprocess`
    
    """
    
    def __init__(self):
        """
        Wraps main methods into :class:`~.Operation` and organizes into a list.
        
        Default postprocessing settings are used as wrapper parameters.
        All wrapped operations are grouped in :list:`~.operations`
        
        Attributes
        ----------
        sampling_rate : int
            Empty parameter that will hold calculated sampling rate value
        operation_op : Operation
            Wrapped postprocessing operation method.
        operations : list
            List holding all wrapped operations.
        """
        
        self.sampling_rate =     None
        self.open_op =           Operation(self, "open", self.open,
                                      {"data_src": "manual"}, True);
        self.cut_op =            Operation(self, "cut", self.cut,
                                      {"ms_to_cut": 0}, True);
        self.debase_op =         Operation(self, "debase", self.debase, 
                                      {"fit_order": 20}, True);
        self.filter_op =         Operation(self, "filter", self.filter,
                                      {"filter_freq": (50, 100, 150, 200,
                                                       250, 300)},
                                      True);
        self.predict_op =        Operation(self, "predict", self.predict,
                                      {'prediction_fill': 'auto',
                                       'prediction_order': 'auto',
                                       'prediction_scan': 'all'},
                                      True);
        self.zerofill_op =       Operation(self, "zerofill", self.zerofill,
                                      {}, True);
        self.apodize_op =        Operation(self, "apodize", self.apodize,
                                      {"decay_c": -0.25}, True);
        self.phasecorrect_op =   Operation(self, "phase_correct", self.phasecorrect,
                                      {"phase_shift": 0}, True)        
        
        self.operations =   [
                                self.open_op, 
                                self.cut_op,
                                self.debase_op,
                                self.filter_op,
                                self.predict_op,
                                self.zerofill_op,
                                self.apodize_op,
                                self.phasecorrect_op
                            ]
    
        
    def update_settings(self, user_settings):  
        """
        Iterates througth list of operations and updates the settings.

        Parameters
        ----------
        user_settings : dict
            dictionary of keyword arguments given by user as custom settings

        Raises
        ------
        TypeError
            User used unknown keyword
        """
        
        for arg in user_settings:
            for i, operation in enumerate(self.operations):
                if arg in operation.parameters:
                    operation.parameters[arg] = user_settings[arg]
                    break
                elif arg == operation.name:
                    operation.enabled = user_settings[arg]
                    break
                elif i == len(self.operations) - 1:
                    raise TypeError(f'{arg} is unknown keyword')
                
    
    def fft(self, y):
        """Function transforms time signal values into spectra via FFT""" 

        vals = np.fft.rfft(y)
        freq = np.fft.rfftfreq(len(y), 1/self.sampling_rate)
        return freq, vals


    def op_by_string_reference(self, operation_ref : str):
        """Returns operation corresponding to string reference"""

        op_refs ={
                 "opened": self.open_op,
                 "cut": self.cut_op,
                 "debased": self.debase_op,
                 "filtered": self.filter_op,
                 "predicted": self.predict_op,
                 "zerofilled": self.zerofill_op,
                 "apodized": self.apodize_op,
                 "phase_corrected": self.phasecorrect_op
                 }        
        try:
            operation = op_refs[operation_ref]
        except:
            raise NameError(f"{operation_ref} is not a valid postprocessing \
                            stage. Use: 'opened', 'cut', 'debased', 'predicted',\
                            'zerofilled', 'apodized', 'phase_corrected'")        
        return operation                            

    def get_results(self, operation_ref : str):
        """Returns PostprocessingStep with results based on string reference"""
        
        look_for = self.op_by_string_reference(operation_ref)
        results = self.postprocessed  

        for result in results:
            if result.operation == look_for:
                return result
        
        raise LookupError(f"{operation_ref} stage is not present int this\
                          Spectrum object")

    def plot(self, stage, mode, rng="all"):
        """
        Plots user defined postprocessing stage, data mode and data range.

        Parameters
        ----------
        stage : {'opened', 'cut', 'debased', 'predicted', 'zerofilled',
                 'apodized', 'phasecorrected'}
            User reference to postprocessing stage to be plotted.
        mode : {'time', 'ftr', 'ftm'}
            Mode of data to be plotted. 
            - 'time' plots a time signal
            - 'ftr' plots a real part of a spectrum.
            - 'ftm' plots a magnitude spectrum.
        rng : iterable of floats or "all", default = "all"
            Data range to be plotted. Must be iterable of floats such
            as (0, 10). Denotes seconds in "time" mode and Hz in "ftm"/"ftr"
            modes.
        """
        stage = self.get_results(stage)
        x, y = self.__get_data_by_mode(stage, mode)
        
        self.__plotpriv(x, y, rng, stage.operation.name)

    def __get_data_by_mode(self, stage, mode):
        """Returns specific data using mode string reference"""
        
        if mode != 'time' and mode != 'ftr' and mode != 'ftm':
            raise TypeError(f"{mode} is not valid mode, must be 'time', \
                            'ftr' or 'ftm'")     
        # Cache data from given stage
        if mode == 'time':
            x, y = stage.x, stage.y
        else:
            x, y = stage.fft[0], stage.fft[1]
        # If magnitude FT requested get absolute value of spectral coeffs.
        if mode == 'ftm':            
            y = np.abs(y)        
            
        return x, y            

    def __plotpriv(self, x, y, rng, *labels):
        """Helper function handling drawing of figures
        
        Parameters
        ----------
        x : iterable
            X axis data.
        y : iterable
            Y axis data.
        rng : iterable of numbers or "all"
            Data range to be plotted.
        *labels : iterable of str
            Plot line labels.
        """
        # By default there are no labels and no data range bounds
        start, end = None, None
        lbl = None
        # If data range bounds defined get corresponding indexes of data arrays.
        if rng != "all":
            try:
                start, end = rng[0], rng[1]
                start, end = self.find_index(x, start, end) 
            except:
                print(f"Range argument must be 'all' or iterable of two\
                      numbers, you have entered {rng}")                    
        # If label was passed in a call, state in in variable
        if len(labels) != 0 and not isinstance(labels[0], str):
            raise TypeError(f'Label must be string. Attempted to pass {labels}')
        elif len(labels) != 0:
            lbl = labels[0]     

        # Plot
        plt.plot(x[start:end], y[start:end], label=lbl)


    def open(self, x, y, fft, data_src="manual"):
        """
        Opens the file or variable containing raw time signal data.
        
        Returns `x`, `y` and their spectrum `fft`. Function also updates
        `Spectrum.sampling_rate` attribute with calculated sampling rate.
        
        Parameters
        ----------
        data_src : str/iterable
            Denotes source of time signal data. Options:
            - "Manual" prompts dialog window.
            - "path/filename.dat" directly accesses file on harddrive
            - iterable (:list:, :tuple:) containing x, y arrays
            
        Notes
        -----
        `x`, `y`, `fft` parameters are ommited from docstring as in case of
        :method:`~.open` they are unused, and only present to make function
        signature compatible with other postprocessing methods.
        """
        
        if isinstance(data_src, str):
            if data_src == 'manual':
                root = tk.Tk()
                root.withdraw()
                data_src = filedialog.askopenfilename()
            data = np.genfromtxt(data_src,
                                 skip_header=1,
                                 skip_footer=1,
                                 names=True,
                                 dtype=None)
        else:
            data = data_src
    
        x, y, *_ = zip(*data)
        x = np.array(x)
        y = np.array(y)
        self.sampling_rate = len(x) / (x[-1] - x[0])        
        fft = ZulfCore.fft(self, y) 
        
        return x, y, fft
    

    def cut(self, x, y, fft, ms_to_cut):
        """
        Function cuts datapoints from the beggining of time signal data.
        
        Appropriate number of points is calculated from `ms_to_cut` paremeter.
        Function returns postprocessed copy of `x`, `y`, `fft`.

        Parameters
        ----------
        x : numpy.ndarray
            Time signal timestamps.
        y : numpy.ndarray
            Time signal values.
        ms_to_cut : int
            Miliseconds of time signal to be removed from the front.
            
        Notes
        -----
        `fft` parameter is ommited from docstring as in case of
        :method:`~.cut` is unused, and only present to make function
        signature compatible with other postprocessing methods.            
        """
        srate = self.sampling_rate

        if ms_to_cut != 0 or ms_to_cut is False:
            cut_points = int(srate * ms_to_cut / 1000)
            x = np.delete(x, range(len(x))[:cut_points])
            y = np.delete(y, range(len(y))[:cut_points])
            x = np.array(x)
            y = np.array(y)
            fft = ZulfCore.fft(self, y)
        
        return x, y, fft
    

    def debase(self, x, y, fft, fit_order):
        """
        Method removes the baseline of time signal.
        
        Polynomial fit of order `fit_order` is substracted from time signal
        data and copy of original data is returned.
        
        Parameters
        ----------
        x : numpy.ndarray
            Time signal timestamps.
        y : numpy.ndarray
            Time signal values.
        fit_order : int
            Order of polynomial fit.
            
        Notes
        -----
        `fft` parameter is ommited from docstring as in case of
        :method:`~.debase` is unused, and only present to make function
        signature compatible with other postprocessing methods.            
        """

        fit_c = np.polynomial.polynomial.polyfit(x, y, fit_order)
        fit_x = np.linspace(x[0], x[-1], len(x))
        fit_y = np.polynomial.polynomial.polyval(fit_x, fit_c)
    
        for i, _ in enumerate(y):
            y[i] = y[i] - fit_y[i]
    
        x = np.array(x)
        y = np.array(y)
        fft = ZulfCore.fft(self, y)
        
        return x, y, fft
    
    
    def filter(self, x, y, fft, filter_freq):
        """
        Method applies IIR filter to time signal and returns its copy.

        Parameters
        ----------
        x : numpy.ndarray
            Time signal timestamps.
        y : numpy.ndarray
            Time signal values.
        filter_freq : iterable of floats
            Iterable containing spectra frequencies to be filtered.

        Notes
        -----
        `fft` parameter is ommited from docstring as in case of
        :method:`~.filter` is unused, and only present to make function
        signature compatible with other postprocessing methods.            
        """

        srate = self.sampling_rate
        
        for i in filter_freq:
            quality_factor = 30
            b_notch, a_notch = sc.signal.iirnotch(i, quality_factor, srate)
            y = sc.signal.filtfilt(b_notch, a_notch, y)
    
        x = np.array(x)
        y = np.array(y)
        fft = ZulfCore.fft(self, y)
        
        return x, y, fft
    
    
    def predict(self, x, y, fft, prediction_fill,
                prediction_order, prediction_scan):
        """
        Method fills missing datapoints at the start of the time signal.
        
        Backward linear prediction is used to construct a front part of time
        signal that had to be removed due to data corruption. Method uses
        Yules-Walker method to get autoregressive-coeffecients. Those are
        inverted to create a backward predictor. Function returns copy of the
        data.

        Parameters
        ----------
        x : numpy.ndarray
            Time signal timestamps.
        y : numpy.ndarray
            Time signal values.
        prediction_fill : int/"auto"
            Number of datapoints to be predicted and prepended.
        prediction_order : int/"auto"
            Order of the prediction filter. Denotes how many previous
            datapoints are used to determine single simulated point. Options:
                -int: order
                -"auto": method will automatically determine the suitable order
                    of the filter. It is recommended to attempt find better 
                    manual setting.
        prediction_scan : int
            How many datapoints are used in Yules-Walker algorithm to determine
            the filter coeffecients. Larger values provide more robust model, 
            however as signal components decay at different ratse their
            representation may be skewed as the sample size is increasing.
            
        Notes
        -----
        `fft` parameter is ommited from docstring as in case of
        :method:`~.predict` is unused, and only present to make function
        signature compatible with other postprocessing methods.              
        """
        # Convinience caching
        fill = prediction_fill
        order = prediction_order
        scan = prediction_scan
        srate = self.sampling_rate
        
        # N of Missing datapoints determined by number of missing time stamps 
        if fill == 'auto':
            fill = int(x[0] * srate)
    
        # Main part of function is nested
        def backward_prediction(data, tofill, order, scan='all'):
            def optimal_order(data, maxorder):
                print('Searching optimal prediction order, may take a moment')
                order = np.arange(1, maxorder)
                rho = [sp.arburg(data, i)[1] for i in order]
                order1 = sp.AIC(len(data), rho, order)
                order2 = sp.AICc(len(data), rho, order)
                order3 = sp.AKICc(len(data), rho, order)
                order4 = sp.CAT(len(data), rho, order)
                order5 = sp.FPE(len(data), rho, order)
                order6 = sp.KIC(len(data), rho, order)
                opt = [order1, order2, order3, order4, order5, order6]
                optimal = [np.argmin(i)+1 for i in opt]
                optimal = int(np.mean(optimal))
                print('Converged to ', optimal, 'degree filter')
                return optimal
    
            #Reverses signal data and calculates AR coeffs.
            data = np.flip(data)
            if scan == 'all':
                if order == 'auto':
                    order = optimal_order(data, tofill)
                autoregressive, *_ = sp.aryule(data, order)
            elif isinstance(scan, int):
                if order == 'auto':
                    order = optimal_order(data[-scan:], tofill)
                autoregressive, *_ = sp.aryule(data[-scan:], order)
            else: raise TypeError('scan keyword must be integer')
    
            # Cache part of time signal needed for prediction 
            predicted = data[-order:]
            # For each point to be predicted it's value is calculated and then
            # appended.
            for _ in range(tofill):
                data_point = sum(predicted[:-order-1:-1]*-autoregressive)
                predicted = np.append(predicted, data_point)
            # All predicted points are appended to rest of data and reversed
            data = np.append(data, predicted[order:])
            return np.flip(data)
    
    
        y = backward_prediction(y, fill, order, scan=scan)
        # Timestamps are generated for new datapoints
        for _ in range(fill):
            x = np.insert(x, 0, x[0] - 1/srate )
    
        x = np.array(x)
        y = np.array(y)
        fft = ZulfCore.fft(self, y)
        
        return x, y, fft


    def zerofill(self, x, y, fft):
        """
        Zero-pads time signal and returns copy of the data.

        Parameters
        ----------
        x : numpy.ndarray
            Time signal timestamps.
        y : numpy.ndarray
            Time signal values.

        Notes
        -----
        `fft` parameter is ommited from docstring as in case of
        :method:`~.zerofill` is unused, and only present to make function
        signature compatible with other postprocessing methods.  
        """        
        
        srate = self.sampling_rate
        
        # Adds zeros equal to total number of datapoints then adds additional
        # zeros such that total number of datapoints is N = 2^k.
        y = ng.process.proc_base.zf_double(np.array(y), 1)
        y = ng.process.proc_base.zf_auto(y)
    
        # Generates timestamps corresponding to the newly added zero datapoints
        tofill = len(y)-len(x)
        t_last = x[-1]+ tofill/srate
        x_fill = np.linspace(x[-1]+1/srate, t_last, tofill)
        x = np.append(x, x_fill)
        
        x = np.array(x)
        y = np.array(y)
        fft = ZulfCore.fft(self, y)
        
        return x, y, fft
    
    
    # Apodization
    def apodize(self, x, y, fft, decay_c):
        """
        Multiplies signal values by decaying exponential and return the copy.

        Parameters
        ----------
        x : numpy.ndarray
            Time signal timestamps.
        y : numpy.ndarray
            Time signal values.
        decay_c : float
            A negative float as c in x*e^c affecting steepness of exponential
            decay.

        Notes
        -----
        `fft` parameter is ommited from docstring as in case of
        :method:`~.apodize` is are unused, and only present to make function
        signature compatible with other postprocessing methods.  
        """
    
        exponential = np.exp(x*decay_c)
        apodized = y * exponential
    
        x = np.array(x)
        y = np.array(apodized)
        fft = ZulfCore.fft(self, y)
        
        return x, y, fft
    
    def interactive_plot(self, freq, vals, x, y, fft):
        """Manages interactive plot for manual phase correction"""
        
        # Figure instantiation
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        line, = plt.plot(freq, vals, lw=2)
        ax.margins(x=0)
    
        # Slider
        axcolor = 'lightgoldenrodyellow'
        axshift = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    
        sshift = Slider(axshift, 'Phase Shift (deg)', -180, 180, valinit=0, valstep=1)
    
        def update(val):
            phs_shown = sshift.val
            line.set_ydata(self.phasecorr(vals, np.full(len(vals), phs_shown/180*np.pi)))
            fig.canvas.draw_idle()
    
    
        sshift.on_changed(update)
    
        # Reset Button
        resetax = plt.axes([0.5, 0.025, 0.1, 0.04])
        btn_reset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    
        def reset(event):
            sshift.reset()
    
        btn_reset.on_clicked(reset)
    
        # Save Button
        saveax = plt.axes([0.8, 0.025, 0.1, 0.04])
        btn_save = Button(saveax, 'Save', color=axcolor, hovercolor='0.975')
    
        # Save button event saves user-adjusted data to corresponding instance
        # :object:`Spectrum.PostprocessingStep`
        def save(event):
            nonlocal vals, x, y, fft, self
            phs = sshift.val
            print(f'Phase of the spectra was shifted by {phs} degrees')
            vals = self.phasecorr(vals, np.full(len(vals), phs/180*np.pi))
           
            self.postprocessed[-1].y = np.fft.irfft(vals)
            self.postprocessed[-1].x = np.array(x)
            self.postprocessed[-1].fft = (freq, vals)
            
            plt.close()

        btn_save.on_clicked(save)
        saveax._btn = btn_save
        plt.show()
   
        # Unadjusted data are returned by default and later rewritten if user
        # succesfully manualy adjusts phase (by `save`).
        return x, y, fft
    
    def phasecorrect(self, x, y, fft, phase_shift):
        """
        Adjusts the phase of the spectra and returns a copy of the data.

        Parameters
        ----------
        x : numpy.ndarray
            Time signal timestamps.
        y : numpy.ndarray
            Time signal values.
        fft : tuple
            Pair of `numpy.ndarray` containing frequency and spectral values.
        phase_shift : float
            Number of degrees the phase of spectra should be adjusted by.
        """
        
        phs = phase_shift
        freq, vals = ZulfCore.fft(self, y)
    
        if phs != 'manual':
            vals = self.phasecorr(vals, np.full(len(vals), phs))
            y = np.fft.irfft(vals)
            x = np.array(x)
            fft = (freq, vals)
            return x, y, fft
        else:
            return ZulfCore.interactive_plot(self, freq, vals, x, y, fft)

    def phasecorr(self, data, ps):
        apod = np.exp(1.0j * ps).astype(data.dtype)
        data_ps = data * apod
        return data_ps
    
    def find_index(self, data, start, end):
        """
        Returns indices of chosen data range.

        Parameters
        ----------
        data : numpy.ndarray
            Frequency bins of the spectra.
        start : float
            Spectral range start frequency.
        end : float
            Spectral range end frequency.

        Returns
        -------
        start : int
            Index of the frequency bin closest to chosen frequency range start.
        end : int
            Index of the frequency bin closest to chosen frequency range end.
        """
        for i in range(len(data)-1):
            if abs(start-data[i]) < abs(start-data[i+1]):
                start = i
                break
            elif i == len(data) - 2:
                start = i + 2
                break
    
        for j in range(len(data)-1):
            if abs(end - data[j]) < abs(end - data[j+1]):
                end = j
                break
            elif j == len(data) - 2:
                end = j + 2
                break
        return start, end


    def iterate(self, pltstage : str, pltmode :str, pltrng, param1, args1,
                *arbargs):
        """
        Method used to iteratively compare different postprocessing settings.
        
        Iterate method allows user to quickly find optimal postrocessing settings
        for given data by passing in the name of parameter and set of arguments to
        be used. Results of postprocessing with each of passed settings are then
        plotted for user to visually compare. It is possible to iterate over two
        parameters at the same time, aiding cases where postprocessing operations
        influence each other's effect. All combinations settings will be plotted as
        per user's choice of data stage, data mode and data range.
    
        Parameters
        ----------
        stage : {'opened', 'cut', 'debased', 'predicted', 'zerofilled',
                 'apodized', 'phasecorrected'}
            User reference to postprocessing stage to be plotted.
        mode : {'time', 'ftr', 'ftm'}
            Mode of data to be plotted. 
            - 'time' plots a time signal
            - 'ftr' plots a real part of a spectrum.
            - 'ftm' plots a magnitude spectrum.
        rng : iterable of floats or "all", default = "all"
            Data range to be plotted. Must be iterable of floats such
            as (0, 10). Denotes seconds in "time" mode and Hz in "ftm"/"ftr"
            modes.
        param1 : {'data_src','cut','ms_to_cut','debase','fit_order','filter',
                  'filter_freq','predict','prediction_fill','prediction_order',
                  'prediction_scan','zerofill','apodize','decay_c','phase_correct',
                  'phase_shift'}
            String reference to postprocessing parameter to be iterated.
        args1 : iterable
            Iterable of viable parameters for param1
        *arbargs : param & args
            Optional second parameter and it's args to be iterated over param1
        """
        from ZULFDSP import Spectrum

    
        # Argument check `pltstage` by passing it to helper method. Return is not
        # needed
        _ = self.op_by_string_reference(pltstage)    
    
        # List of all possible settings that can be iterated in Spectrum class
        settings = ['data_src',
                    'cut',
                    'ms_to_cut',
                    'debase',
                    'fit_order',
                    'filter',
                    'filter_freq',
                    'predict',
                    'prediction_fill',
                    'prediction_order',
                    'prediction_scan',
                    'zerofill',
                    'apodize',
                    'decay_c',
                    'phase_correct',
                    'phase_shift'  
                    ]
    
        # Argument checks                        
        name_err = f" is not a valid setting that can be iterated.\
                    Valid iterable settings : {settings}"
        
        if param1 not in settings:
            raise NameError(param1, name_err) 
    
        # See whether we iterate over two parameters. If so do argument check.
        param2 = None
        args2 = None
        if len(arbargs) != 0:
            assert len(arbargs) == 2, "Use of arbitrary arguments only allows for\
                entering another iterable parameter 'param2' and it's set of\
                arguments to iterate over 'args2' "                    
            param2 = arbargs[0]
            args2 = arbargs[1]
            if param2 not in settings:
                raise NameError(param2, name_err) 
            
        # Create copy of original kwargs to iterate settings on
        kwargs_temp = dict(self.kwargs)
        # For each argument on given parameter
        for i, argument1 in enumerate(args1):
            kwargs_temp[param1] = argument1
            # For each argument of second beiing iterated (if exists)
            if param2 != None:                     
                for j, argument2 in enumerate(args2):
                    kwargs_temp[param2] = argument2
                    # Reconstruct temporary spectrum w. new parameters
                    spectrum_temp = Spectrum(**kwargs_temp)
    
                    # Save the results 
                    result = spectrum_temp.get_results(pltstage)
                    if j == 0:
                        results2 = [result]
                    else:
                        results2.append(result)
                # Save results in 2D list
                if i == 0:
                    results = [results2]      
                else:
                    results.append(results2)  
            
            # If only one parameter to iterate over
            else:
                spectrum_temp = Spectrum(**kwargs_temp)
                
                result = spectrum_temp.get_results(pltstage)
                if i == 0:
                    results = [result]
                else:
                    results.append(result)
    
        # Create figure that will hold multiple plots
        plt.figure()
        plt.xlabel('Time [s]' if pltmode == 'time' else 'Freq [Hz]')
        plt.ylabel('Amplitude [V]')
        
        # Plot the gathered results.
        for i, result1 in enumerate(results):
            # If only one parameter is iterated all plots are in single premade
            # figure.
            if param2 == None:
                name = param1 + ' ' + str(args1[i])
                x, y = self.__get_data_by_mode(result1, pltmode)            
                self.__plotpriv(x, y, pltrng, name)
                plt.legend()
            
            # If iterating over two parameters where param1 has n elements in args1
            # and param2 has m elements in args 2, ther will be n figures created
            # each containing m plots,
            else:
                plt.title(param1 + ' ' + str(args1[i]))
                for j, result2 in enumerate(result1):
                    name = param2 + ' ' + str(args2[j])
                    x, y = self.__get_data_by_mode(result2, pltmode)                 
                    self.__plotpriv(x, y, pltrng, name)
                    plt.legend()
                # If this was not last figure, create a new figure.
                if result1 is not results[-1]:
                    plt.figure()
                    plt.xlabel('Time [s]' if pltmode == 'time'
                               else 'Freq [Hz]')
                    plt.ylabel('Amplitude [V]')
    
