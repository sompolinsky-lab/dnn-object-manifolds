function tuning_function=calc_randomization_single_neurons(full_tuning_function, global_preprocessing)
    [N_NEURONS, N_SAMPLES, N_OBJECTS] = size(full_tuning_function);
    if global_preprocessing == 1
        tuning_function = reshape(full_tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]);
        tuning_mean = nanmean(tuning_function , 2);
        tuning_function = bsxfun(@minus, tuning_function, tuning_mean);
        a =  nanstd(tuning_function, [], 2); a(a==0) = 1;
        tuning_function = bsxfun(@rdivide, tuning_function, a);
        tuning_mean = bsxfun(@rdivide, tuning_mean, a);
        assert_warn(norm(nanmean(tuning_function.^2, 2)-(N_SAMPLES*N_OBJECTS-1)/(N_SAMPLES*N_OBJECTS),inf) < 1e-10);
        tuning_function = bsxfun(@plus, tuning_function, tuning_mean);
        tuning_function = reshape(tuning_function, [N_NEURONS, N_SAMPLES, N_OBJECTS]);
    elseif global_preprocessing == 2
        tuning_function = reshape(full_tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]);
        tuning_function = bsxfun(@minus, tuning_function, nanmean(tuning_function , 2));
        [pU, pS, ~] = svd(tuning_function, 0); pS=diag(diag(pS));
        tuning_function = (pS\pU'*tuning_function).*sqrt(N_SAMPLES*N_OBJECTS-1);
        assert(norm(nanmean(tuning_function.^2, 2)-(N_SAMPLES*N_OBJECTS-1)/(N_SAMPLES*N_OBJECTS),inf) < 1e-10);
        assert(norm(tuning_function*tuning_function'/(N_SAMPLES*N_OBJECTS-1)-eye(N_NEURONS), inf) < 1e-6);
        tuning_function = reshape(tuning_function, [N_NEURONS, N_SAMPLES, N_OBJECTS]);
    elseif global_preprocessing == 3
        % Project into a subspace where the centers are decorelated
        Xc = reshape(nanmean(full_tuning_function, 2), [N_NEURONS, N_OBJECTS]);
        [~, cS, cV] = svd(Xc, 0); 
        I = eye(N_NEURONS);
        W=(I-Xc/(Xc'*Xc)*Xc')+Xc*cV*(cS^-3)*cV'*Xc';
        assert(all(isnan(W(:))==0));
        assert(all(size(W) == [N_NEURONS, N_NEURONS]));
        tuning_function = reshape(W*reshape(full_tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]), [N_NEURONS, N_SAMPLES, N_OBJECTS]);
        decorelated_centers = squeeze(nanmean(tuning_function, 2));
        Cc = decorelated_centers'*decorelated_centers;
        assert_warn(norm(Cc-eye(N_OBJECTS), inf) < 1e-8, sprintf('Deviation from I: %1.3e', norm(Cc-eye(N_OBJECTS), inf)));
    elseif global_preprocessing == 4 || global_preprocessing == 5
        % Project into a subspace where the centers are decorelated
        Xc = reshape(nanmean(full_tuning_function, 2), [N_NEURONS, N_OBJECTS]);
        [cU, cS, cV] = svd(Xc, 0); 
        Var = diag(diag(Xc'*Xc));
        I = eye(N_NEURONS);
        W=(I-Xc/(Xc'*Xc)*Xc')+cU*(Var^0.5)*cV*(cS^-2)*cV'*Xc';
        assert(all(isnan(W(:))==0));
        assert(all(size(W) == [N_NEURONS, N_NEURONS]));
        tuning_function = reshape(W*reshape(full_tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]), [N_NEURONS, N_SAMPLES, N_OBJECTS]);
        decorelated_centers = squeeze(nanmean(tuning_function, 2));
        Cc = decorelated_centers'*decorelated_centers;
        assert_warn(norm(Cc-Var, inf) < 1e-6, sprintf('Deviation from I: %1.3e', norm(Cc-Var, inf)));
        if global_preprocessing == 5
            global_mean = nanmean(reshape(tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]), 2);
            tuning_function = bsxfun(@minus, tuning_function, global_mean);
        end
    elseif global_preprocessing == 6
        % Project into the subspace spanned by the center where they are decorelated
        Xc = reshape(nanmean(full_tuning_function, 2), [N_NEURONS, N_OBJECTS]);
        [cU, cS, ~] = svd(Xc, 0);
        reduceW=cS\cU';
        assert(all(size(reduceW) == [N_OBJECTS, N_NEURONS]));
        tuning_function = reshape(reduceW*reshape(full_tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]), [N_OBJECTS, N_SAMPLES, N_OBJECTS]);
        reduced_centers = squeeze(nanmean(tuning_function, 2));
        deviationI = norm(reduced_centers'*reduced_centers-eye(N_OBJECTS), inf);
        assert_warn(deviationI < 1e-10, sprintf('Deviation from I in centers span: %1.3e', deviationI));
        %global_mean = nanmean(reshape(tuning_function, [N_OBJECTS, N_SAMPLES*N_OBJECTS]), 2);
        %tuning_function = bsxfun(@minus, tuning_function, global_mean);
    elseif global_preprocessing == 7
        % Project into a subspace where the centers are decorelated
        Xc = reshape(nanmean(full_tuning_function, 2), [N_NEURONS, N_OBJECTS]);
        [cU, ~, ~] = svd(Xc, 0);
        reduceW=cU';
        assert(all(size(reduceW) == [N_OBJECTS, N_NEURONS]));
        tuning_function = reshape(reduceW*reshape(full_tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]), [N_OBJECTS, N_SAMPLES, N_OBJECTS]);

        global_mean = nanmean(reshape(tuning_function, [N_OBJECTS, N_SAMPLES*N_OBJECTS]), 2);
        tuning_function = bsxfun(@minus, tuning_function, global_mean);
    elseif global_preprocessing == 8
        % Project into a subspace where the centers are decorelated
        Xc = reshape(nanmean(full_tuning_function, 2), [N_NEURONS, N_OBJECTS]);
        Var = diag(diag(Xc'*Xc));
        [cU, cS, cV] = svd(Xc, 0);
        reduceW=Var^0.5*cV/cS*cU';
        assert(all(size(reduceW) == [N_OBJECTS, N_NEURONS]));
        tuning_function = reshape(reduceW*reshape(full_tuning_function, [N_NEURONS, N_SAMPLES*N_OBJECTS]), [N_OBJECTS, N_SAMPLES, N_OBJECTS]);
        reduced_centers = squeeze(nanmean(tuning_function, 2));
        assert(norm(reduced_centers'*reduced_centers-Var, inf) < 1e-10);

        global_mean = nanmean(reshape(tuning_function, [N_OBJECTS, N_SAMPLES*N_OBJECTS]), 2);
        tuning_function = bsxfun(@minus, tuning_function, global_mean);
    else
        tuning_function = full_tuning_function;
    end
end
