function A = compute_A(t0, t1)
    A = zeros(8);

    for col = 1:8
        A(1, col) = t0^(col-1);
        A(5, col) = t1^(col-1);
        if col > 1
            A(2, col) = (col-1)*t0^(col-2);
            A(6, col) = (col-1)*t1^(col-2);
        end
        if col > 2
            A(3, col) = (col-1)*(col-2)*t0^(col-3);
            A(7, col) = (col-1)*(col-2)*t1^(col-3);
        end
        if col > 3
            A(4, col) = (col-1)*(col-2)*(col-3)*t0^(col-4);
            A(8, col) = (col-1)*(col-2)*(col-3)*t1^(col-4);
        end
    end
end