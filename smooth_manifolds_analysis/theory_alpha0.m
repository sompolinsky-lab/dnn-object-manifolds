function a=theory_alpha0(kappa)
    if isnan(kappa)
        a=nan;
        return;
    end
    if kappa>100
        a=kappa^-2;
        return;
    end
    Dt = @(t) exp(-0.5*t.^2)/sqrt(2*pi);
    f = @(t) Dt(t).*(t+kappa).^2;
    I = integral(f, -kappa, inf);
    a = I^-1;
end
%x=0:0.1:2;
%for i=1:length(x)
%y(i)=theory_alpha0(x(i));
%end
%plot(x, y)
%ylim([0, 2]);
