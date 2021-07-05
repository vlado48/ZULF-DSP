# .iterate() plots data postprocessed with different settings
# plot settings (pltstage, pltmode, pltrng) are itentical to .plot() method
# parameters.
# Some postprocessing setting fun1:(parameter of spectrum()) will be
# iterated over set of viable settings inpset1: tuple.
# Data of each iteration are stored in multi d np.array and subsequently
# plotted
# Since each postprocessing run uses same variables to hold data, to
# avoid original data (when object was constructed) being rewritten they
# are saved and copied back at the end of all iterations.
# it is also possible to iterate over two functions by adding arbitrary
# arguments fun2 and inpset2. In this case for each iteration of fun1
# there will be plot with all iterations of fun2
def iterate(self, pltstage, pltmode, pltrng, fun1, inpset1, *arg):

    # Dict of possible iterables with the order of their corresponding
    # postprocessing operation. Iterables always are parameters of
    # spectrum() class constructor
    iterable_ord = {'directory': 0, 'cut': 1,
                    'debase': 2, 'fitorder': 2,
                    'filter': 3, 'filterfreq': 3,
                    'predict': 4, 'prediction_fill': 4,
                    'prediction_order': 4, 'prediction_scan': 4,
                    'zero_pad': 5,
                    'apodize': 6, 'decay_c': 6,
                    'phase_correct': 7, 'phase_shift': 7
                    }

    settings = self.settings
    params = self.params

    # Save original settings & results.
    # Each function creates its own set of attributes xn, yn, fftn storing
    # data, where n is position in order
    # this is important because user will often times want to see
    # changes in data after individual operations.
    settings_orig = dict(self.settings)
    params_orig = dict(self.params)
    rslts_orig = {}
    for i in range(8):  # there is 8 functions in postprocessing
        attrx = 'x' + str(i)
        attry = 'y' + str(i)
        attrfft = 'fft' + str(i)
        if hasattr(self, attry):
            rslts_orig[str(i)] = ((getattr(self, attrx),
                                   getattr(self, attry)),
                                  getattr(self, attrfft))

    # Argument check for plotting argument, fun1 and inpset1. Then check
    # if *aargs fun2 and inpset2 were used, if so check correctness and
    # change var iterables from = 1 to  = 2
    if pltstage not in self.__operations:
        raise TypeError(f'{pltstage} is not proper stage of \
                        postprocessing')
    if pltmode != 'time' and pltmode != 'ftr' and pltmode != 'ftm':
        raise TypeError(f'{pltmode} is not proper form of data. Choose \
                        time, ftr or ftm')
        # There can only be two additional arguments (fun2, inpset2)
    if len(arg) != 0 and len(arg) != 2:
        raise TypeError('Incorrect arguments for second iterable. \
                        Arguments must have form fun2, settings2')
    elif len(arg) != 0:
        fun2, inpset2 = arg
        iterables = 2       # in this case numer of iterables is = 2
        if not isinstance(inpset2, list)\
                and not isinstance(inpset2, tuple):
            raise TypeError(f'Settings of {fun2} must be in form of list \
                            or tuple of correct setting type')
        if fun2 not in iterable_ord:
            raise TypeError(f'{fun2} is not correct iterable of \
                            postprocessing')

        if not isinstance(inpset1, list) and not isinstance(inpset1, tuple):
            raise TypeError(f'Settings of {fun1} must be in form of list or\
                            tuple of correct setting type')
        if fun1 not in iterable_ord:
            raise TypeError(f'{fun1} is not correct iterable of \
                            postprocessing')

    # Var iterables seys with how many iterators we work with on this fun
    # call. Earlier_iter says which of those, fun1 or fun2 is first in
    # postprocessing sequence.
    iterables = 1
    earlier_iter = 1

    # Iteratively change settings, postprocess and save the data from
    # requested stage (car pltstage). Determine which of fun to be iterated
    # is earlier in sequence, set its position in sequence as var 'start'.

    if 'fun2' not in locals() or iterable_ord[fun1] < iterable_ord[fun2]:
        start = iterable_ord[fun1]
    else:
        start = iterable_ord[fun2]
        earlier_iter = 2

    for i, stg1 in enumerate(inpset1):    # For each setting inputted
        if fun1 in settings:              # change setting or params dict.
            settings[fun1] = stg1
        else:
            params[fun1] = stg1

        if iterables == 2:                       # Change settings again if
            for j, stg2 in enumerate(inpset2):   # 2nd iterable exists
                if fun2 in settings:
                    settings[fun2] = stg2
                else:
                    params[fun2] = stg2
                self.__postprocess(start)       # Run postprocessing

                # Pick correct data form and save them to list
                if pltmode != 'time':
                    data = self.__lastxy(pltstage, 'ft')
                else:
                    data = self.__lastxy(pltstage, 'time')
                if j == 0:
                    results2 = [data]
                else:
                    results2.append(data)
            if i == 0:
                results = [results2]      # Create main list of result
            else:
                results.append(results2)  # Append list of secondary itera-
                # tions to primary list.

        else:
            self.__postprocess(start)

            if pltmode != 'time':
                data = self.__lastxy(pltstage, 'ft')
            else:
                data = self.__lastxy(pltstage, 'time')

            if i == 0:
                results = [data]
            else:
                results.append(data)

    # Create figures with data plot from each iteration. If there are two
    # iterables there will be x figures, where x is len(inpset1)
    plt.figure()
    plt.xlabel('Time [s]' if pltmode == 'time' else 'Freq [Hz]')
    plt.ylabel('Amplitude [V]')
    for i, it1 in enumerate(results):
        if iterables == 1:
            name = fun1 + ' ' + str(inpset1[i])
            self.__plotpriv(it1[0], it1[1] if pltmode != 'ftm' else
                            np.abs(it1[1]), pltrng, name)
            plt.legend()
        else:
            plt.title(fun1 + ' ' + str(inpset1[i]))
            for j, it2 in enumerate(it1):
                name = fun2 + ' ' + str(inpset2[j])
                self.__plotpriv(it2[0], it2[1] if pltmode != 'ftm' else
                                np.abs(it2[1]), pltrng, name)
                plt.legend()
            if it1 is not results[-1]:
                plt.figure()
                plt.xlabel('Time [s]' if pltmode == 'time'
                           else 'Freq [Hz]')
                plt.ylabel('Amplitude [V]')

    # Delete results held from iterating / replace by original results
    for i in range(start, 8):
        n = str(i)
        if hasattr(self, 'x' + n):
            delattr(self, 'x' + n)
            delattr(self, 'y' + n)
            delattr(self, 'fft' + n)
        if n in rslts_orig:
            setattr(self, 'x' + n, rslts_orig[n][0][0])
            setattr(self, 'y' + n, rslts_orig[n][0][1])
            setattr(self, 'fft' + n, rslts_orig[n][1])
