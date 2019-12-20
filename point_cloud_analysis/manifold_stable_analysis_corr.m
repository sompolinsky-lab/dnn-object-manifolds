%% Compute theoretical capacity of the general manifold.
%% Author: SueYeon Chung. Feb 28. 2018. 
%% Efficient calcaulation of MFT capacity, R_M, D_M. 
%% Take into account the correlations between the centers
%% Additional measures to make projection to center nullspace stable added. 2018.02.08. 
function [output] = manifold_stable_analysis_corr(XtotT, options)
kappa = options.kappa; 
n_t=options.n_t;
flag_NbyM = options.flag_NbyM; 
center_scale = options.center_scale;
%function [a_Mfull_vec, a_M_vec, R_M_vec,D_M_vec, res_coeff0, K0] = manifold_stable_analysis_corr(XtotT, kappa, n_t, flag_NbyM)
% rng(1); 
% XtotT: cell file. Each cell is M_i by N matrix, where M_i is number of
% samples in i_{th} manifold, and N is the feature dimension. 

P=length(XtotT); 
for ii=1:P
    if ~flag_NbyM 
        Xtot{ii}=XtotT{ii}';    
    else
        Xtot{ii}=XtotT{ii}; 
    end
    N=size(Xtot{ii},1); 
end
clear XtotT; 
Xori=[]; 
for pp=1:P
    M_vec(pp)=size(Xtot{pp},2); %% #(samples) vec. 
    Xori=[Xori Xtot{pp}];  %% All data.
end

M0=sum(M_vec); % Total number of samples. 
X0=mean(Xori,2); %Global Mean 
clear Xori; 
centers=nan(N,P);
centers_old = nan(N,P);
for pp=1:P
    Xtot0_old{pp}=(Xtot{pp}-repmat(X0, [1 M_vec(pp)]));   % Previously X, data-global mean
    centers_old(:,pp)=mean(Xtot0_old{pp},2); 
    centers(:,pp) = mean(Xtot0_old{pp},2)*center_scale;
    Xtot0{pp}= (Xtot0_old{pp}- centers_old(:,pp)) + centers(:,pp); 
end
clear Xtot;

%% Center correlation analysis 

[UU,SS,VV]=svd(centers-repmat(mean(centers,2),[1 P]));  %figure; plot(diag(S))
ll=diag(SS); 
maxK=max(find(cumsum(ll.^2./(sum(ll.^2)))<0.95))+10; 
MAX_ITER=20000; 
%MAX_ITER=100;
N_REPEATS = 1; % Number of repetitions to find the most stable soln at each iteration of K. 

[norm_coeff, norm_coeff_vec, Proj, V1_mat, res_coeff, res_coeff0] = fun_FA(centers, maxK, MAX_ITER, N_REPEATS); 
[res_coeff_opt, KK]=min(res_coeff);
K0=KK;
%K0=0; 
fprintf('Optimal K: %d.\n', K0)
V11=Proj*V1_mat{K0}; % Projection vector into the low rank structure 

Xr0_ns_norm = nan(1,P);
for ii=1:P
    M=M_vec(ii);     
    Xr=squeeze(Xtot0{ii}); % Data for each manifold, global mean subtracted (center not normalized yet)            
    Xr_ns=Xr-V11*(V11'*Xr); % Project to null space of center subspace 
    Xr0_ns=mean(Xr_ns,2); % Centers of manifolds in null space 
    Xr0_ns_norm(ii) = norm(Xr0_ns);
    Xrr_ns= (Xr_ns-repmat(Xr0_ns,[1 M]))/Xr0_ns_norm(ii); % cent-normalized data in the center null-space 
    XtotInput{ii}=Xrr_ns; 

end
clear Xtot0; 

%% First make (D+1) dimensional data 
for ii=1:P
    tic
    S_r=XtotInput{ii}; 
    [D,m]= size(S_r);   % D-dimensional data, 

    %% Project data into the smaller space (not necessary for small dataset) 
    if D>m
        [Q,R]=qr(S_r,0); % Q is [D, m]
        S_rNew=Q'*S_r;  
        S_rOld=S_r; D_old=D; 
        S_r=S_rNew; 
        [D,m]= size(S_r);
        %fprintf('Reduced: D=%d, m=%d.\n', D,m)
    end 
    sD=zeros(D,m); 
    for kk=1:(D+1)
        if kk<D+1
            sD(kk,:)=S_r(kk,:);
        else
            sc = 1; 
        end
    end
    %% Make data D+1 dimensional, adding center dimension 
    sD1_0 = [sD; repmat(sc, [1 m])]; % (D+1) by m 
    sD1 = sD1_0/sc; 
    [a_Mfull, a_M, R_M,D_M] = each_manifold_analysis_D1(sD1, kappa, n_t);
    R_M_vec(ii)=R_M; 
    D_M_vec(ii)=D_M; 
    a_M_vec(ii)=a_M; 
    a_Mfull_vec(ii)=a_Mfull; 
    simtime(ii)=toc; 
    fprintf('%d th manifold: D=%d, m=%d, D_M=%.2f, R_M=%.2f, a_M=%.2f, %.2f sec., norm=%1.3f\n',...
    ii, D,m, D_M, R_M, a_M, simtime(ii), Xr0_ns_norm(ii))
end
fprintf('Average of %d manifold: <D_M>=%.2f, <R_M>=%.2f, 1/<1/a_M>=%.2f.\n',P, ...
    mean(D_M_vec), mean(R_M_vec), 1./mean(1./a_M_vec))
fprintf('STD of %d manifold: std(D_M)=%.2f, std(R_M)=%.2f, std(a_M)=%.2f.\n',P, std(D_M_vec), std(R_M_vec), std(a_M_vec))
output.a_Mfull_vec = a_Mfull_vec; 
output.a_M_vec = a_M_vec; 
output.R_M_vec = R_M_vec; 
output.D_M_vec = D_M_vec; 
output.res_coeff0 = res_coeff0; 
output.K0 = K0; 
end

function [a_Mfull, a_M, R_M,D_M] = each_manifold_analysis_D1(sD1, kappa, n_t)
[D1, m] = size(sD1);   % D+1-dimensional data
D = D1 - 1; 
sc = 1; 
c_hat=zeros(1,D+1)'; c_hat(end,1)=1; 
t_vec = randn(D+1,n_t);
[ss, gg] = maxproj(t_vec, sD1, sc);

s_all=zeros(D+1,n_t);
v_f_all=zeros(D+1,n_t); 
for jj=1:n_t
    if gg(jj)+kappa < 0 % Interior Points 
        v_f=t_vec(:,jj); 
        s_f=ss(:,jj); 
    else
        eps0=1e-8;
          [v_f, alpha, vminustsq, exitflag]=compute_v_allpt(t_vec(:,jj),ss(:,jj),eps0, kappa, sD1); 

         t0=t_vec(:,jj)'*c_hat; 
         v0=v_f'*c_hat; 
         if norm(t_vec(:,jj)-v_f)<1e-8 % Interior. 
            v_f=t_vec(:,jj); 
            s_f=ss(:,jj);
         else
            lambda=sum(alpha); 
            l_vec(jj)=lambda; 
            s_f=(t_vec(:,jj)-v_f)/lambda; 

            vD=v_f(1:D,1);tD=t_vec(1:D,jj);sD=s_f(1:D,1);
            ov=tD'*sD/(norm(tD)*norm(sD));
            norm(s_f(1:D,1));
            maxvs=max(v_f'*sD1);
         end
    end
    s_all(:,jj)=s_f; 
    v_f_all(:,jj)=v_f; 
    a_all(jj)=1./(norm(v_f-t_vec(:,jj)).^2); 
end

s0 = mean(s_all,2); 
ds0 = s_all-repmat(s0,[1 n_t]); 
ds = ds0(1:end-1,:)./repmat(s_all(end,:),[D 1]); 
R_M = sqrt(mean(sum(ds(:, :).^2,1)));
MW_M = mean(abs(sum(t_vec(1:end-1,:)'.*ds(:, :)',2)));  
tD_vec=t_vec(1:end-1,:); 
sD_vec= s_all(1:end-1,:); 
t_hat_vec = tD_vec./repmat(sqrt(sum(tD_vec.^2,1)),[D 1]);
s_hat_vec = sD_vec./repmat(sqrt(sum(sD_vec.^2,1)),[D 1]);
D_M = D*(mean(sum(t_hat_vec.*s_hat_vec,1)))^2;
k_M = R_M.*sqrt(D_M);
a_M=alphaB(0, R_M, D_M); 
a_Mfull = 1/mean((max(sum(t_vec.*s_all,1)+kappa,0)).^2 ./ sum(s_all.^2,1)); 
end

function [ output] = alphaB(kappa, radius, d)
k=kappa; R=radius; 
fun = @(r, t) A(k, R, r, t).*exp(-t.^2./2)./sqrt(2*pi).*P_d(r, d); 

L=50; 
alphainv = integral2(fun, 0, L, -L, L);
%alphainv = integral2(fun, 0, Inf, -Inf, Inf);
%  output = dblquad(fun, 0, L, -L, L);
%output = dblquad(fun, -Inf, Inf, -Inf, Inf)
output= 1./alphainv;     
    function out_p = P_d(r, d)
        %% Original p_d
        out_p = 2.^(1-d/2).*r.^(d-1).*exp(-0.5*r.^2)./gamma(d/2);      
        %% Approximate p_d
        %out_p = exp(-0.5.*d.*log(2)+d.*log(r)-0.5*r.^2-0.5*d*log(d./2)+d./2); 
    end 
    function out_r = A(k, R, r, t)         
        out_r = (R.*r-t+k).^2./(R.^2+1).*((t-(k-r./R))>0).*(((k+R.*r)-t)>0) ...
                + ((t-k).^2+r.^2).*(((k-r./R)-t)>0); 
    end 
end

function [v_f, alpha,vminustsq, exitflag]= compute_v_allpt(tt, sDi, eps0, kappa, sD1)
% tt=[D+1,1]; S=[D+1, m]; eps=small. 

[D1, m] = size(sDi); D=D1-1; 
Tk=sD1; 
sc= sDi(end,1); 
flag_con=1;  
k=1; vminustsqk=10000;  
Fk_old=vminustsqk; 
[v_k, vt_k , exitflag, alphak, vminustsqk]= minimize_vt_sq(tt, Tk, kappa); 
    
v_f=v_k; 
vminustsq=vminustsqk;
alpha=alphak;
end

function [s0, gt] = maxproj(t_vec, sD1, sc)
    n_t=size(t_vec,2);
    D1= size(t_vec,1); D=D1-1; 
    m= size(sD1,2); 
    for i=1:n_t
        tt= t_vec(:,i);
        [gt0, imax]= max(tt(1:D,1)'*sD1(1:D,:));
        sr = sD1(1:D,imax); 
        s0(:,i)=[sr; sc];
        gt(i)=tt'*s0(:,i); 
    end
end

function [v_f, vt_f , exitflag, alphar, normvt2] = minimize_vt_sq(t, sD1, kappa)
% normvt is the objective function. 
    [D1, m]=size(sD1); D=D1-1; 
    H=eye(D1,D1);
    f=-t; 
    Aineq= sD1'; %[m by D1];
    bineq= -kappa*ones(m,1); 
    
    %% cplexqp options 
%     tolerance=1e-10; 
%     options = cplexoptimset('cplex');
%     options.simplex.tolerances.feasibility = tolerance;
%     options.simplex.tolerances.optimality = tolerance;
%     options.display = 'off';
%     options.MaxIter = 1e25;
%     options.qpmethod = 2; 
%     options =  optimoptions('Display','off');
    options = optimset('Display', 'off');
    %%
%     [v_f, vt_f,exitflag, output, alpha0] = cplexqp(H,f,Aineq,bineq,[],[],[],[],[],options);
     [v_f, vt_f,exitflag, output, alpha0] = quadprog(H,f,Aineq,bineq,[],[],[],[],[], options);
    normvt=(vt_f+0.5*sum(t.^2))*2;
    normvt2= sum((v_f-t).^2);
    alphar=alpha0.ineqlin; % Vector of lagrange coefficients for each data 
end 


function [norm_coeff, norm_coeff_vec, P, V1_mat, res_coeff, res_coeff0] = fun_FA(centers, maxK, MAX_ITER, N_REPEATS)

[N_NEURONS, N_OBJECTS]=size(centers);
Ks = 1:maxK; 
use_optimal_recovered_solution = true;
mean_residual_corrcoef = zeros(1, 1+length(Ks));
real_solution_overlap = zeros(1, 1+length(Ks));
X=centers';
opts = struct;
opts.record = 0;    % no print out
opts.mxitr  = MAX_ITER;  % max number of iterations
opts.gtol = 1e-6;   % stop control for the projected gradient
opts.xtol = 1e-6;   % stop control for ||X_k - X_{k-1}||
opts.ftol = 1e-8;   % stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
                    % usually, max{xtol, gtol} > ftol
% Reduce global mean and Gram-Schmit into the relevant basis
global_mean = mean(X, 1);
Xb = bsxfun(@minus, X, global_mean);
xbnorm = sqrt(sum(Xb.^2, 2));
[q, r] = qr(Xb', 0);   
N_NEURONS= N_OBJECTS-1;
P = q(:,1:N_NEURONS);
X = Xb*P;
%Vreal = bsxfun(@minus, Vreal, global_mean)*P;
%% Before subtracting 
X0 = X;          % [P, N], Data after extracting the low rank structure 
xnorm = sqrt(sum(X0.^2, 2));
C0 = X0*X0' ./ (xnorm*xnorm');
res_coeff0 = (sum(abs(C0(:)))-N_OBJECTS)/N_OBJECTS/(N_OBJECTS-1);
%%
V1 = [];
for ik=1:length(Ks)
    T = tic;
    k = Ks(ik);
    best_stability = 0;
    best_V1 = [];
    for ir=1:N_REPEATS
        s = randn(N_OBJECTS, 1);    % [P, 1]
        V0 = [s'*X; V1'];           % [K, N]
        [V0, ~] = qr(V0', 0); assert(all(size(V0) == [N_NEURONS, k]));
        [V1tmp, output]= OptStiefelGBB(V0, @square_corrcoeff_full_cost, opts, X);
    %     assert_warn(output.itr < MAX_ITER, sprintf('Max iterations reached at k=%d (%s)', k, output.msg))
    %    assert(output.itr < MAX_ITER, sprintf('Max iterations reached at k=%d (%s)', k, output.msg))
        cost_after = square_corrcoeff_full_cost(V1tmp, X);
        assert(abs(output.fval - cost_after) < 1e-8);
        assert(norm(V1tmp'*V1tmp-eye(k))< 1e-10);
        X0 = X-(X*V1tmp)*V1tmp';          % [P, N], Data after extracting the low rank structure 
        stability = min(sqrt(sum(X0.^2, 2)) ./ sqrt(sum(X.^2, 2)));
        if stability > best_stability
            best_stability = stability;
            best_V1 = V1tmp;
        end
        if N_REPEATS > 1
            fprintf(' [%d] cost=%1.3f stability=%1.3f\n', ir, cost_after, stability);
        end
    end
    V1 = best_V1;
    %Vhat = [Vhat; v];
    %calc_subspace_overlap(V, Vhat)
    %overlap = calc_subspace_overlap(V1', Vreal);
    X0 = X-(X*V1)*V1';          % [P, N], Data after extracting the low rank structure 
    xnorm = sqrt(sum(X0.^2, 2));
    C0 = X0*X0' ./ (xnorm*xnorm');
    current_cost = (sum(abs(C0(:)))-N_OBJECTS)/N_OBJECTS/(N_OBJECTS-1);
    fprintf(' K=%d mean=%1.3f (took %1.1f sec, %d iterations)\n', ...
        k, current_cost, toc(T), output.itr);
    mean_residual_corrcoef(ik+1) = current_cost;
    V1_mat{ik}=V1; 
    C0_mat{ik}=C0;
   
    norm_coeff{ik}=xnorm./xbnorm;
    norm_coeff_vec(ik)= mean(xnorm./xbnorm); 
    res_coeff(ik)=current_cost; 
    if (ik>4) && (res_coeff(ik)>res_coeff(ik-1)) && (res_coeff(ik-1)>res_coeff(ik-2)) ...
            && (res_coeff(ik-2)>res_coeff(ik-3))
        fprintf('Optimal K0 found. \n')
        break;
    end
end
    
% P*V1_mat{iK} % Projection into null space of low rank [N Nnull]
zKs = [0, Ks];
end

function [X, out]= OptStiefelGBB(X, fun, opts, varargin)
%-------------------------------------------------------------------------
% curvilinear search algorithm for optimization on Stiefel manifold
%
%   min F(X), S.t., X'*X = I_k, where X \in R^{n,k}
%
%   H = [G, X]*[X -G]'
%   U = 0.5*tau*[G, X];    V = [X -G]
%   X(tau) = X - 2*U * inv( I + V'*U ) * V'*X
%
%   -------------------------------------
%   U = -[G,X];  V = [X -G];  VU = V'*U;
%   X(tau) = X - tau*U * inv( I + 0.5*tau*VU ) * V'*X
%
%
% Input:
%           X --- n by k matrix such that X'*X = I
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptStiefelGBB(X0, @fun, opts, data1, data2);
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%   
% Output:
%           X --- solution
%         Out --- output information
%
% -------------------------------------
% For example, consider the eigenvalue problem F(X) = -0.5*Tr(X'*A*X);
%
% function demo
% 
% function [F, G] = fun(X,  A)
%   G = -(A*X);
%   F = 0.5*sum(dot(G,X,1));
% end
% 
% n = 1000; k = 6;
% A = randn(n); A = A'*A;
% opts.record = 0; %
% opts.mxitr  = 1000;
% opts.xtol = 1e-5;
% opts.gtol = 1e-5;
% opts.ftol = 1e-8;
% 
% X0 = randn(n,k);    X0 = orth(X0);
% tic; [X, out]= OptStiefelGBB(X0, @fun, opts, A); tsolve = toc;
% out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
% fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(XT*X-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norm(X'*X - eye(k), 'fro') );
% 
% end
% -------------------------------------
%
% Reference: 
%  Z. Wen and W. Yin
%  A feasible method for optimization with orthogonality constraints
%
% Author: Zaiwen Wen, Wotao Yin
%   Version 1.0 .... 2010/10
%-------------------------------------------------------------------------


%% Size information
if isempty(X)
    error('input X is an empty matrix');
else
    [n, k] = size(X);
end

if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-6;
    end
else
    opts.xtol = 1e-6;
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-6;
    end
else
    opts.gtol = 1e-6;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-12;
    end
else
    opts.ftol = 1e-12;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
   if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
   end
else
    opts.rho = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
   if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
   end
else
    opts.eta = 0.2;
end

% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
   if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
   end
else
    opts.gamma = 0.85;
end

if isfield(opts, 'tau')
   if opts.tau < 0 || opts.tau > 1e3
        opts.tau = 1e-3;
   end
else
    opts.tau = 1e-3;
end

% parameters for the  nonmontone line search by Raydan
if ~isfield(opts, 'STPEPS')
    opts.STPEPS = 1e-10;
end

if isfield(opts, 'nt')
    if opts.nt < 0 || opts.nt > 100
        opts.nt = 5;
    end
else
    opts.nt = 5;
end

if isfield(opts, 'projG')
    switch opts.projG
        case {1,2}; otherwise; opts.projG = 1;
    end
else
    opts.projG = 1;
end

if isfield(opts, 'iscomplex')
    switch opts.iscomplex
        case {0, 1}; otherwise; opts.iscomplex = 0;
    end
else
    opts.iscomplex = 0;
end

if isfield(opts, 'mxitr')
    if opts.mxitr < 0 || opts.mxitr > 2^20
        opts.mxitr = 1000;
    end
else
    opts.mxitr = 1000;
end

if ~isfield(opts, 'record')
    opts.record = 0;
end


%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
rho  = opts.rho;
STPEPS = opts.STPEPS;
eta   = opts.eta;
gamma = opts.gamma;
iscomplex = opts.iscomplex;
record = opts.record;

nt = opts.nt;   crit = ones(nt, 3);

invH = true; if k < n/2; invH = false;  eye2k = eye(2*k); end

%% Initial function value and gradient
% prepare for iterations
[F,  G] = feval(fun, X , varargin{:});  out.nfe = 1;  
GX = G'*X;

if invH
    GXT = G*X';  H = 0.5*(GXT - GXT');  RX = H*X;
else
    if opts.projG == 1
        U =  [G, X];    V = [X, -G];       VU = V'*U;
    elseif opts.projG == 2
        GB = G - 0.5*X*(X'*G);
        U =  [GB, X];    V = [X, -GB];       VU = V'*U;
    end
    %U =  [G, X];    VU = [GX', X'*X; -(G'*G), -GX];   
    %VX = VU(:,k+1:end); %VX = V'*X;
    VX = V'*X;
end
dtX = G - X*GX;     nrmG  = norm(dtX, 'fro');
    
Q = 1; Cval = F;  tau = opts.tau;

%% Print iteration header if debug == 1
if (opts.record == 1)
    fid = 1;
    fprintf(fid, '----------- Gradient Method with Line search ----------- \n');
    fprintf(fid, '%4s %8s %8s %10s %10s\n', 'Iter', 'tau', 'F(X)', 'nrmG', 'XDiff');
    %fprintf(fid, '%4d \t %3.2e \t %3.2e \t %5d \t %5d	\t %6d	\n', 0, 0, F, 0, 0, 0);
end

%% main iteration
for itr = 1 : opts.mxitr
    XP = X;     FP = F;   GP = G;   dtXP = dtX;
     % scale step size

    nls = 1; deriv = rho*nrmG^2; %deriv
    while 1
        % calculate G, F,
        if invH
            [X, infX] = linsolve(eye(n) + tau*H, XP - tau*RX);
        else
            [aa, infR] = linsolve(eye2k + (0.5*tau)*VU, VX);
            X = XP - U*(tau*aa);
        end
        %if norm(X'*X - eye(k),'fro') > 1e-6; error('X^T*X~=I'); end
        if ~isreal(X) && ~iscomplex ; error('X is complex'); end
        
        [F,G] = feval(fun, X, varargin{:});
        out.nfe = out.nfe + 1;
        
        if F <= Cval - tau*deriv || nls >= 5
            break;
        end
        tau = eta*tau;          nls = nls+1;
    end  
    
    GX = G'*X;
    if invH
        GXT = G*X';  H = 0.5*(GXT - GXT');  RX = H*X;
    else
        if opts.projG == 1
            U =  [G, X];    V = [X, -G];       VU = V'*U;
        elseif opts.projG == 2
            GB = G - 0.5*X*(X'*G);
            U =  [GB, X];    V = [X, -GB];     VU = V'*U; 
        end
        %U =  [G, X];    VU = [GX', X'*X; -(G'*G), -GX];
        %VX = VU(:,k+1:end); % VX = V'*X;
        VX = V'*X;
    end
    dtX = G - X*GX;    nrmG  = norm(dtX, 'fro');
    
    S = X - XP;         XDiff = norm(S,'fro')/sqrt(n);
    tau = opts.tau; FDiff = abs(FP-F)/(abs(FP)+1);
    
    if iscomplex
        %Y = dtX - dtXP;     SY = (sum(sum(real(conj(S).*Y))));
        Y = dtX - dtXP;     SY = abs(sum(sum(conj(S).*Y)));
        if mod(itr,2)==0; tau = sum(sum(conj(S).*S))/SY; 
        else tau = SY/sum(sum(conj(Y).*Y)); end    
    else
        %Y = G - GP;     SY = abs(sum(sum(S.*Y)));
        Y = dtX - dtXP;     SY = abs(sum(sum(S.*Y)));
        %alpha = sum(sum(S.*S))/SY;
        %alpha = SY/sum(sum(Y.*Y));
        %alpha = max([sum(sum(S.*S))/SY, SY/sum(sum(Y.*Y))]);
        if mod(itr,2)==0; tau = sum(sum(S.*S))/SY;
        else tau  = SY/sum(sum(Y.*Y)); end
        
        % %Y = G - GP;
        % Y = dtX - dtXP;
        % YX = Y'*X;     SX = S'*X;
        % SY =  abs(sum(sum(S.*Y)) - 0.5*sum(sum(YX.*SX)) );
        % if mod(itr,2)==0;
        %     tau = SY/(sum(sum(S.*S))- 0.5*sum(sum(SX.*SX)));
        % else
        %     tau = (sum(sum(Y.*Y)) -0.5*sum(sum(YX.*YX)))/SY;
        % end
        
    end
    tau = max(min(tau, 1e20), 1e-20);
    
    if (record >= 1)
        fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e  %3.2e  %2d\n', ...
            itr, tau, F, nrmG, XDiff, FDiff, nls);
        %fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e (%3.2e, %3.2e)\n', ...
        %    itr, tau, F, nrmG, XDiff, alpha1, alpha2);
    end
    
    crit(itr,:) = [nrmG, XDiff, FDiff];
    mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
    %if (XDiff < xtol && nrmG < gtol ) || FDiff < ftol
    %if (XDiff < xtol || nrmG < gtol ) || FDiff < ftol
    %if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol 
    %if ( XDiff < xtol || FDiff < ftol ) || nrmG < gtol     
    if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])  
        if itr <= 2
            ftol = 0.1*ftol;
            xtol = 0.1*xtol;
            gtol = 0.1*gtol;
        else
            out.msg = 'converge';
            break;
        end
    end
    
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
 end

if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.feasi = norm(X'*X-eye(k),'fro');
if  out.feasi > 1e-13
    X = MGramSchmidt(X);
    [F,G] = feval(fun, X, varargin{:});
    out.nfe = out.nfe + 1;
    out.feasi = norm(X'*X-eye(k),'fro');
end

out.nrmG = nrmG;
out.fval = F;
out.itr = itr;

end

function [cost, gradient] = square_corrcoeff_full_cost(V, X)
    [P, N] = size(X);
    K = size(V,2);
    assert(all(size(V) == [N, K])); 
    %assert(norm(V'*V-eye(K))  < 1e-10);
    % Calculate cost
    C = X*X';                       % [P x P]
    c = X*V;                        % [P x K]
    c0 = diag(C) - sum(c.^2, 2);    % [P x 1]
    Fmn = (C-c*c').^2./(c0*c0');    % [P x P]
    cost = sum(Fmn(:))/2;
    % Calculate gradient if needed
    if nargout > 1
        X1 = reshape(X, [1, P, N]);
        X2 = reshape(X, [P, 1, N]);
        C1 = reshape(c, [P, 1, 1, K]);
        C2 = reshape(c, [1, P, 1, K]);
        Gmni =      - bsxfun(@times, (C-c*c')./(c0*c0'), bsxfun(@times, C1, X1));
        Gmni = Gmni - bsxfun(@times, (C-c*c')./(c0*c0'), bsxfun(@times, C2, X2));
        Gmni = Gmni + bsxfun(@times, (C-c*c').^2./(c0*c0').^2, bsxfun(@times, c0, bsxfun(@times, C2, X1)));
        Gmni = Gmni + bsxfun(@times, (C-c*c').^2./(c0*c0').^2, bsxfun(@times, c0', bsxfun(@times, C1, X2)));
        gradient = reshape(sum(sum(Gmni, 1), 2), [N, K]);
    end
end

function V = MGramSchmidt(V)
[n,k] = size(V);

for dj = 1:k
    for di = 1:dj-1
        V(:,dj) = V(:,dj) - proj(V(:,di), V(:,dj));
    end
    V(:,dj) = V(:,dj)/norm(V(:,dj));
end
end


%project v onto u
function v = proj(u,v)
v = (dot(v,u)/dot(u,u))*u;
end