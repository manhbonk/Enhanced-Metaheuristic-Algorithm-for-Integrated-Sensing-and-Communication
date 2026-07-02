function [Fronts, Rank] = fastNonDominatedSort(Pop)

    N = numel(Pop);
    
    %==============================================================
    % Initialization
    %==============================================================
    Rank = inf(N,1);
    
    DominationCount = zeros(N,1);
    
    DominatedSet = cell(N,1);
    
    Fronts = cell(N,1);
    
    %==============================================================
    % First Front
    %==============================================================
    front1 = zeros(1,N);
    front1Count = 0;
    
    for p = 1:N
    
        Sp = zeros(1,N);
        spCount = 0;
    
        obj_p = Pop(p).Obj;
    
        for q = 1:N
    
            if p == q
                continue;
            end
    
            obj_q = Pop(q).Obj;
    
            if dominates(obj_p, obj_q)
    
                spCount = spCount + 1;
                Sp(spCount) = q;
    
            elseif dominates(obj_q, obj_p)
    
                DominationCount(p) = DominationCount(p) + 1;
    
            end
    
        end
    
        DominatedSet{p} = Sp(1:spCount);
    
        if DominationCount(p) == 0
    
            Rank(p) = 1;
    
            front1Count = front1Count + 1;
            front1(front1Count) = p;
    
        end
    
    end
    
    Fronts{1} = front1(1:front1Count);
    
    %==============================================================
    % Remaining Fronts
    %==============================================================
    frontNo = 1;
    
    while true
    
        currentFront = Fronts{frontNo};
    
        if isempty(currentFront)
            break;
        end
    
        nextFront = zeros(1,N);
        nextCount = 0;
    
        for ii = 1:numel(currentFront)
    
            p = currentFront(ii);
    
            Sp = DominatedSet{p};
    
            for jj = 1:numel(Sp)
    
                q = Sp(jj);
    
                DominationCount(q) = DominationCount(q) - 1;
    
                if DominationCount(q) == 0
    
                    Rank(q) = frontNo + 1;
    
                    nextCount = nextCount + 1;
                    nextFront(nextCount) = q;
    
                end
    
            end
    
        end
    
        if nextCount == 0
            break;
        end
    
        Fronts{frontNo+1} = nextFront(1:nextCount);
    
        frontNo = frontNo + 1;
    
    end
    
    Fronts = Fronts(1:frontNo);

end

%==================================================================
% Pareto Dominance (Minimization)
%==================================================================
function flag = dominates(obj1, obj2)

    flag = all(obj1 <= obj2) && any(obj1 < obj2);

end