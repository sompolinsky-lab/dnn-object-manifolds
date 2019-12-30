function magnitude=calc_affine_transform_magnitude(transform)
    if strcmp(char(class(transform)),'affine2d')
        T = transform.T;
    else
        assert(strcmp(char(class(transform)),'double'));
        T = transform;
    end
    assert(all(T(:,3)==[0;0;1]));
    A=[1,1;1,-1;-1,1;-1,-1];
    v=T(1:6)-[1,0,0,0,1,0];
    magnitude=sqrt(mean((A*v(1:2)'+v(3)).^2)+mean((A*v(4:5)'+v(6)).^2));
end
