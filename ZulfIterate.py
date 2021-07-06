
def iterate(self, pltstage : str, pltmode :str, pltrng, param1, args1, *arbargs):
    
    # Cache the operation that user wants to plot data of
    plotted_op = self.op_by_string_reference(pltstage)    

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
                spectrum_temp = Spectrum(kwargs_temp)

                # Save the results 
                result = self.get_results(pltstage)
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
            spectrum_temp = Spectrum(kwargs_temp)
            
            result = self.get_results(pltstage)
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

