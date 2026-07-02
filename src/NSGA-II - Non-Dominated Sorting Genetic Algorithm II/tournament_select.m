function idx = tournament_select(Pop, K)

    N = numel(Pop);
    
    %----------------------------------------------------------
    % Random candidates
    %----------------------------------------------------------
    cand = randi(N,[K,1]);
    
    winner = cand(1);
    
    for i = 2:K
    
        challenger = cand(i);
    
        %----------------------------------------------
        % Compare Pareto Rank
        %----------------------------------------------
        if Pop(challenger).Rank < Pop(winner).Rank
    
            winner = challenger;
    
        elseif Pop(challenger).Rank == Pop(winner).Rank
    
            %------------------------------------------
            % Compare Crowding Distance
            %------------------------------------------
            if Pop(challenger).Distance > Pop(winner).Distance
    
                winner = challenger;
    
            elseif Pop(challenger).Distance == Pop(winner).Distance
    
                % Random tie-break
                if rand < 0.5
                    winner = challenger;
                end
    
            end
    
        end
    
    end
    
    idx = winner;

end