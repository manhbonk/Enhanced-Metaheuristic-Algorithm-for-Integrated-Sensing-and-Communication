function Pop = assignRankAndDistance(Pop)

    N = numel(Pop);
    
    %------------------------------------------------------------
    % Fast Non-dominated Sorting
    %------------------------------------------------------------
    [Fronts, Rank] = fastNonDominatedSort(Pop);
    
    for i = 1:N
        Pop(i).Rank = Rank(i);
    end
    
    %------------------------------------------------------------
    % Crowding Distance
    %------------------------------------------------------------
    for f = 1:length(Fronts)
    
        idx = Fronts{f};
    
        if isempty(idx)
            continue;
        end
    
        Dist = crowdingDistance(Pop(idx));
    
        for k = 1:length(idx)
            Pop(idx(k)).Distance = Dist(k);
        end
    
    end

end