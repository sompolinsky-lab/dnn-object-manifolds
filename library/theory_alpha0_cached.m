function alpha=theory_alpha0_cached(kappa)
    global CACHED_THEORY_ALPHA0;
    global CACHED_THEORY_ALPHA0_KAPPAS;
    if isempty(CACHED_THEORY_ALPHA0) || isempty(CACHED_THEORY_ALPHA0_KAPPAS)
        T=tic;
        CACHED_THEORY_ALPHA0_KAPPAS = -50:0.01:100;
        CACHED_THEORY_ALPHA0 = zeros(size(CACHED_THEORY_ALPHA0_KAPPAS));
        for i=1:length(CACHED_THEORY_ALPHA0_KAPPAS)
            CACHED_THEORY_ALPHA0(i) = theory_alpha0(CACHED_THEORY_ALPHA0_KAPPAS(i));
        end
        fprintf('Created alpha0 cache (took %1.1f sec)\n', toc(T));
    end
    I = kappa>100;
    alpha = nan(size(kappa));
    alpha(I) = kappa(I).^-2;
    alpha(~I) = interp1(CACHED_THEORY_ALPHA0_KAPPAS, CACHED_THEORY_ALPHA0, kappa(~I), 'linear', inf);
    assert_warn(all(isfinite(alpha(:)) | isnan(alpha(:))), sprintf('Infinite values. Kappa range [%1.3e, %1.3e]', min(kappa(:)), max(kappa(:))));
    assert_warn(all(alpha(:)>0 | isnan(alpha(:))), sprintf('Negative values. Kappa range [%1.3e, %1.3e]', min(kappa(:)), max(kappa(:))));    
end
