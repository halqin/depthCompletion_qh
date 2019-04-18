function  morphin = morph_mask(morphin, morplabel, aniin, threshold)
inst_label =morplabel ~= 0;
resi = inst_label .*( morphin -  morplabel);
th_index = find(abs(resi)>threshold);
% inst_resi = resi ~= 0;

% index_ = find(inst_resi); % find inst_resi larger than threshold 
morphin(th_index) = aniin(th_index);% extract optimized value according to morph nonzero position
                                                  % map the ani optimized value to morphological only 
end 