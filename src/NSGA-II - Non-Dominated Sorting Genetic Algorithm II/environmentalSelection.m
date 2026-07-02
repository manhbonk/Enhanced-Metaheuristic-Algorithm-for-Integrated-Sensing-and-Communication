function Pop = environmentalSelection(MergePop,Npop)

    Ranks = [MergePop.Rank];
    maxRank = max(Ranks);
    
    Pop = repmat(MergePop(1),Npop,1);
    
    count = 0;
    
    for r = 1:maxRank
    
        idx = find(Ranks==r);
    
        if isempty(idx)
            continue;
        end
    
        if count + numel(idx) <= Npop
    
            Pop(count+1:count+numel(idx)) = MergePop(idx);
    
            count = count + numel(idx);
    
        else
    
            Dist = [MergePop(idx).Distance];
    
            [~,ord] = sort(Dist,'descend');
    
            remain = Npop-count;
    
            Pop(count+1:Npop) = MergePop(idx(ord(1:remain)));
    
            break;
    
        end
    
    end

end