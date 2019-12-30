function assert_warn(b, message, varargin)
    if ~b
        fprintf(['Warning: ', message, '\n'], varargin{:});
    end
end
