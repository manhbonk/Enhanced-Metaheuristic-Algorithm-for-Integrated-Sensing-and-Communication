function Distance = crowdingDistance(Pop)

    N = numel(Pop);
    
    %==============================================================
    % Special cases
    %==============================================================
    Distance = zeros(N,1);
    
    if N == 0
        return;
    end
    
    if N == 1
        Distance(1) = inf;
        return;
    end
    
    if N == 2
        Distance(:) = inf;
        return;
    end
    
    %==============================================================
    % Number of objectives
    %==============================================================
    numObj = numel(Pop(1).Obj);
    
    %==============================================================
    % Crowding Distance
    %==============================================================
    for m = 1:numObj
    
        % Extract m-th objective
        Obj = zeros(N,1);
        for i = 1:N
            Obj(i) = Pop(i).Obj(m);
        end
    
        % Sort by objective
        [ObjSorted, order] = sort(Obj);
    
        % Boundary solutions
        Distance(order(1)) = inf;
        Distance(order(end)) = inf;
    
        objMin = ObjSorted(1);
        objMax = ObjSorted(end);
    
        % Avoid division by zero
        if objMax == objMin
            continue;
        end
    
        % Interior solutions
        for k = 2:N-1
    
            if isinf(Distance(order(k)))
                continue;
            end
    
            Distance(order(k)) = Distance(order(k)) + ...
                (ObjSorted(k+1) - ObjSorted(k-1)) / (objMax - objMin);
    
        end
    
    end

end