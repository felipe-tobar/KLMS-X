def delaySignal(signal,delay):
    # This function generates a delayed signal from an input "signal" and a delay order. The signal can be one or multi-channel
    import numpy as np
    if np.ndim(signal) > 1:
        r,N = np.shape(signal)
        delayedSignal = signal[:,delay:]
        s = (r,N-delay,delay)   # r x n-d x d
        regresor = np.zeros(s)
        for i in range(N-delay):
            regresor[:,i,:] = signal[:,i:i+delay] # r x d
    else:
        numRows = 1
        numCols = len(signal)
        delayedSignal = signal[delay:numCols:1]
        s = (delay,numCols-delay)    
        regresor = np.zeros(s)
        for i in range(delay,numCols-1):
            regresor[:,i+1-delay] = signal[i-delay:i:1]
        regresor = regresor.T
    return [delayedSignal,regresor]

def gaussian_kernel(x,y,*args):
    # This function applies the gaussian kernel between the inputs x and y, using kernel parameter given by *args
    import numpy as np
    # c x r x d
    gamma = args
    if np.ndim(y) == 3:
        dX = x-y
        KG = np.exp(-gamma[0]*np.sum(np.sum(dX**2, axis=2), axis=1))
    else:
        #c,d = np.shape(y)
        if len(gamma) == 1:
            dX = x-y
            KG = np.zeros(c)
            for i,dx in enumerate(dX):
                KG[i] = np.exp(-gamma[0]*np.sum(dx**2))
        else:
            gamma_aux = gamma[0]
            KG = np.zeros(np.shape(x[:,:,0].T))
            for j in range(len(y)):
                for i in range(len(gamma_aux)):
                    KG[i,j] = gaussian_kernel(x[j,i,:],y[j,i,:],gamma_aux[i])
    return KG

def gaussian_kernel_multi_channel(x,y,*args):
    # This function applies the gaussian kernel for the multi-channel implementation of KLMS-X
    import numpy as np
    gamma = args
    gamma = np.asarray(gamma)
    dX = x-y
    KG = np.exp(np.sum(-gamma*np.sum(dX**2, axis=2),axis=1))
    return KG

def adaptive_filter(x,y,pred_step,kernel_true,gram_matrix,kernel_function,kernel_params,delay,learning_rate,sparsification,\
                    sparsification_params,adaptive_kernel,adaptive_uncertainty,adaptive_params,ard_true,presence_step):
    # This function applies the adaptive filter defined by the user. Currently implements the LMS and KLMS filter with one-
    # channel or multi-channel signals
    
    # x: input signal
    # y: output signal
    # pred_step: number of steps the filter extrapolates before using a new sample /INT
    # kernel_true: usage of kernel /TRUE OR FALSE
    # gram_matrix: in case of constant kernel, precalculated gram_matrix (deprecated)
    # kernel_function: in case of variant kernel, function to calculate kernel
    # delay: degree of autorregresive model
    # learning_rate: learning rate for stepest descent of gradient for model update
    # sparsification: sparsification rule /NOVELTY
    # adaptive_kernel: usage of adaptive kernel /TRUE OR FALSE
    # adaptive_uncertainty: usage of estimation of uncertainty /TRUE OR FALSE
    # adaptive_params: initial uncertainty, initial kernel parameters
    # ard_true: usage of ARD /TRUE OR FALSE (deprecated)
    import numpy as np
    import time
    import numpy.matlib as npm
    if sparsification == 'Novelty':
        dist1 = sparsification_params[0]
        dist2 = sparsification_params[1]
        dist3 = sparsification_params[2]
    
    d = delay
    if len(learning_rate) > 1:
        eta_alpha = learning_rate[0]
        eta_gamma = learning_rate[1]
        eta_sigma = learning_rate[2]
        eta_pres = learning_rate[3]
    else:
        eta_alpha = learning_rate
        eta_gamma = 0
        eta_sigma = 0
        eta_pres = 0
    
    if np.ndim(y) > 1:
        # multichannel algorithm
        #print("Multi-channel")
        [N,r] = np.shape(y)
        
    else:
        # one channel algorithm
        #print("One-channel")
        N = len(y)
        r = 1
    y_estimate = np.zeros((N,r))
    error = np.zeros((N,r))
    sigma_noise_history = np.ones(N)
    gamma_history = np.zeros((N,r))
    runtime = np.zeros((N,r))
    presence = np.zeros((N,N))
    dummy_counter = 0
    center_count = np.zeros(N)
    count_center = 0
    count_center_prune = 0
    ind_displacement = 0
    eps = 1e-8
    
    if kernel_true == False:
        # print("Kernel off")
        x_input = x # Regressor matrix of dimensions N-d x d x r
        alpha = np.zeros((r*d,1))
        alpha_history = np.zeros((N,r,r*d))
        alpha_history[0,:,:] = alpha.T
    else:
        # print("Kernel on")
        x_input = x # Regressor matrix of dimensions N-d x N-d
        alpha = 1*y[0] # weight initialisation
        alpha_history = np.zeros((N,r,N)) # matrix with dimensions N-d x r x c
        alpha_history[0,:,0] = alpha
        if np.ndim(y) > 1:
            center = x_input[:,0,:]
        else:
            center = x_input[0] # saves the support vector
            center = center[:,np.newaxis].T
        center_history = np.zeros((N,r,d)) # matrix of dimensions c x r x d
        center_history[0,:,:] = center
        center_index = np.zeros(N) # variable which saves the index of the new center
        center_index[0] = 1
        if kernel_function == gaussian_kernel:
            gamma_history[0] = kernel_params
        if adaptive_kernel == False and np.ndim(gram_matrix)>1:
            y_estimate[0] = alpha*gram_matrix[0][0]
        else:
            y_estimate[0] = alpha
        center_num = sum(center_index) # the center num equals the index of the accepted centers in the array
        center_num = int(center_num)
        if adaptive_uncertainty:
            sigma_noise_history[0] = adaptive_params[0]
        error[0] = y[0] - y_estimate[0] 
    count_step = 1
#--------------------------------------------------------------------------------------------------------------------#                    
    # KLMS & LMS Filter
    if kernel_true: # KLMS filter
        if (adaptive_kernel == False and np.ndim(gram_matrix)>1): # gram matrix known (currently in development)
            for i in range(N-1):
                if np.ndim(y) > 1:
                    x_k = x_input[:,i,:]
                else:
                    x_k = x_input[i]
                    x_k = x_k[:,np.newaxis].T
                alpha = alpha_history[i,:,:center_num]
                current_kernel = gram_matrix[i]*center_index # uses the value of the accepted kernels
                current_kernel = current_kernel[current_kernel>0] # filter out the unaccepted ones
                norm_x = norm_input(x_k,center_history[:center_num,:,:])
                fact_aux = current_kernel*norm_x
                y_estimate[i+1] = np.dot(alpha.T,current_kernel) # estimation between the dot product of weights and kernel evaluations
                norm_kernel = np.linalg.norm(current_kernel) # kernel norm
                last_kernel = current_kernel[-1] # last kernel evaluated
                if count_step < pred_step:
                    error[i+1] = y_estimate[i] - y_estimate[i+1]
                    count_step += 1
                else:
                    error[i+1] = y[i+1] - y_estimate[i+1] # estimation error
                    count_step = 1
                error_norm = np.linalg.norm(error[i+1]) # estimation error norm
                if error_norm > 1e6:
                    print('Divergence Break')
                    break
                err = error[i]
                err = err[:,np.newaxis]
                current_kernel = np.tile(current_kernel,(r,1)) # kernel evaluation propagates through different channels (c x r)
                current_kernel = np.sum(current_kernel,1) # addition of different channels
                alpha = alpha + eta_alpha*(1/sigma_noise_history[i])*(error[i]*current_kernel)/(eps + norm_kernel) #actualizacion pesos
                gamma_history[i+1] = gamma_history[i] + eta_gamma*err*np.dot(alpha.T,fact_aux)
                sigma_noise_history[i+1] = sigma_noise_history[i] + eta_sigma*(r/sigma_noise_history[i] - (1/sigma_noise_history[i]**3)*err**2)
                
                sparse2 = np.linalg.norm(error[i+1]*y[i+1])
                # Sparsification criterion
                if np.max(current_kernel) <= dist1: # kernel distance criterion
                    if sparse2 >= dist2: # error criterion
                        if np.ndim(y) > 1:
                            center = x_input[:,i,:]
                        else:
                            center = x_input[i] # save support vector
                            center = center[:,np.newaxis].T
                        center_history[i,:,:] = center
                        center_index[i] = 1 # update of the center index array with new center added
                        center_num = sum(center_index) # update of current support vectors quantity
                        center_num = int(center_num)
                        alpha_history[i+1,:,:center_num-1] = alpha # initial condition of new weight
                else:
                    alpha_history[i+1,:,:center_num] = alpha # keep the weights from the last iteration
#--------------------------------------------------------------------------------------------------------------------#                    
        elif (adaptive_kernel == False and np.ndim(gram_matrix) == 1): # kernel function given by user
            gram_matrix = np.eye((N))
            for i in range(N-1):
                s_time = time.time()
                if np.ndim(y) > 1:
                    x_k = x_input[:,i,:]
                else:
                    x_k = x_input[i]
                    x_k = x_k[:,np.newaxis].T
                kernel_eval = gram_matrix_row(x_k,center_history[:center_num,:,:],kernel_function,kernel_params) # gram matrix row calculation
                gram_matrix[i,:center_num] = kernel_eval # gram matrix update
                alpha = alpha_history[i,:,:center_num]
                current_kernel = kernel_eval
                last_kernel = current_kernel[-1] # last kernel evaluation
                norm_kernel = np.linalg.norm(current_kernel) # norm of the used kernels
                norm_x = norm_input(x_k,center_history[:center_num,:,:])
                fact_aux = current_kernel*norm_x
                y_estimate[i+1] = np.dot(alpha,current_kernel) # estimation realized with the dot product between weights and kernel evaluations
                if count_step < pred_step:
                    error[i+1] = y_estimate[i] - y_estimate[i+1]                  
                    count_step += 1
                else:
                    error[i+1] = y[i+1] - y_estimate[i+1] # estimation error
                    count_step = 1
                error_norm = np.linalg.norm(error[i+1]) # estimation error norm
                if error_norm > 1e6:
                    print('Divergence Break')
                    break
                err = error[i]
                err = err[:,np.newaxis]
                nrm_ker = norm_kernel
                alpha = alpha + eta_alpha*(1/sigma_noise_history[i])*(err*current_kernel)/(eps + nrm_ker) #weight update
                gamma_history[i+1] = gamma_history[i] + np.sum(eta_gamma*err*alpha*fact_aux,axis=1)
                sigma_noise_history[i+1] = sigma_noise_history[i] + eta_sigma*(r/sigma_noise_history[i] - (1/sigma_noise_history[i]**3)*np.sum(err**2))
                sparse2 = np.linalg.norm(error[i+1]*y[i+1])
                
                runtime[i] = time.time() - s_time
                # Sparsification criterion
                if np.max(current_kernel) <= dist1: # kernel distance criterion
                    if sparse2 >= dist2: # error criterion
                        center = x_k
                        center_history[center_num,:,:] = center
                        center_index[i] = 1 # index update of the new support vector relative to the position in the gram matrix
                        center_num = sum(center_index) # update of the new quantity of support vectors
                        center_num = int(center_num)
                        alpha_history[i+1,:,:center_num-1] = alpha # initialisation of the new weight
                else:
                    alpha_history[i+1,:,:center_num] = alpha # keep the weights from the last iteration
                # Presence criterion
                if i%presence_step == 0 and eta_pres > 0:
                    pres = (1-eta_pres)*presence[i-1,:center_num] + eta_pres*gram_matrix[i,:center_num]
                    presence[i,:center_num] = pres
                    ind_dummy = np.where(pres > -1)
                    ind_del1 = np.where(pres < dist3)
                    ind_pres2 = np.where(pres == 0)
                    ind_del = np.setdiff1d(ind_del1,ind_pres2) 
                    ind_pres = np.setdiff1d(ind_dummy,ind_del)
                    cindex_1 = np.where(center_index == 1)
                    center_index_aux = center_index[:]
                    center_num_aux = center_num
                    center_index_aux[cindex_1[0][ind_del]] = 0
                    center_num_aux = sum(center_index_aux)
                    center_num_aux = int(center_num_aux)
                    if len(ind_del) > 0 and center_num > 1:
                        dummy_counter += 1
                        center_history[:center_num_aux,:,:] = center_history[ind_pres,:,:]
                        alpha_history[i+1,:,:center_num_aux] = alpha_history[i+1,:,ind_pres].T
                        center_index = center_index_aux[:]
                        if center_num > center_num_aux:
                            count_center_prune += 1
                        else:
                            count_center += 1
                        center_num = center_num_aux
                        ind_displacement = min(cindex_1)
                center_count[i] = center_num                   

#--------------------------------------------------------------------------------------------------------------------#
    else: # LMS filter
        for i in range(N-1): 
            s_time = time.time()
            if np.ndim(y) > 1:
                x_k = x_input[:,i,:].T
            else:
                x_k = x_input[i]
            alpha = alpha_history[i,:,:]
            if np.ndim(y) > 1:
                x_k_rep = npm.repmat(x_k,2,1)
                est = x_k_rep.T*alpha
                y_estimate[i+1] = np.sum(est,1) # estimation computed through the dot product of weights and regressors
            else:
                y_estimate[i+1] = np.dot(alpha,x_k)
            if count_step < pred_step:
                error[i+1] = y_estimate[i] - y_estimate[i+1]
                count_step += 1
            else:
                error[i+1] = y[i+1] - y_estimate[i+1] # estimation error
                count_step = 1
            err = error[i]
            err = err[:,np.newaxis]
            if np.ndim(y) > 1:
                alpha = alpha + eta_alpha*(1/sigma_noise_history[i])*err*x_k_rep.T/(eps + np.linalg.norm(x_k_rep)) # weight update rule
            else:
                alpha = alpha + eta_alpha*(1/sigma_noise_history[i])*error[i]*x_k.T/(eps + np.linalg.norm(x_k)) # weight update rule
            sigma_noise_history[i+1] = sigma_noise_history[i] + eta_sigma*(r/sigma_noise_history[i] - (1/sigma_noise_history[i]**3)*np.sum(err**2))
            alpha_history[i+1,:,:] = alpha
            runtime[i] = time.time() - s_time
    if kernel_true:
        count_arr = [count_center,count_center_prune]
        return [y_estimate,alpha_history,error,sigma_noise_history,gamma_history,center_num,center_count,center_history,center_index,gram_matrix,runtime,count_arr]
    else:
        return [y_estimate,alpha_history,error,sigma_noise_history,runtime]
    
def gram_matrix_row(x,y,kernel_function,*args):
    # This function computes the gram matrix row using the inputs x and y, the kernel function and the kernel parameters *args
    import numpy as np
    c,r,d = np.shape(y) # x has shape r x d
    x_rep = np.tile(x,(c,1,1)) # kernel value replicated to each center, resulting in dimension (c x r)
    G_row = kernel_function(x_rep,y,*args)
    return G_row

def norm_input(x,y):
    # This function returns the norm between two matrices, replicating vector x along a new axis, generating a matrix of the same dimensions as y
    import numpy as np
    c,r,d = np.shape(y)
    x_rep = np.tile(x,(c,1,1)) # kernel value replicated to each center, resulting in dimension (c x r)
    NI = np.linalg.norm(x_rep-y)
    return NI

def xcorr(x,y,lag):
    # This function computes the cross-correlation between the signals x and y given a lag
    import numpy as np
    aux = 0
    result = np.zeros(lag)
    for i in range(lag):
        if i == 0:
            aux = np.cov(x,y)[0,1]
        else:
            aux = np.cov(x[i:lag],y[:-i+lag])[0,1]
        result[i] = aux/(np.std(x[i:lag])*np.std(y[:-i+lag]))
    return result

def autocorr_test(timeseries):
    # This function computes the autocorrelation of a signal
    import numpy as np
    timeseries_copy = np.copy(timeseries)
    mean = np.mean(timeseries_copy)
    timeseries_copy -= np.mean(timeseries_copy)
    autocorr_f = np.correlate(timeseries_copy, timeseries_copy, mode='full')
    temp = autocorr_f[autocorr_f.size/2:]/autocorr_f[autocorr_f.size/2]
    iact = []
    iact.append(sum(autocorr_f[autocorr_f.size/2:]/autocorr_f[autocorr_f.size/2]))
    return temp