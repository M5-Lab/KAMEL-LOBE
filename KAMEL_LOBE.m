%%% S. Arman Ghaffarizadeh & Gerald J. Wang %%%
%%% Getting over the Hump with KAMEL-LOBE: Kernel-Averaging Method to Eliminate Length-Of-Bin Effects in Radial Distribution Functions %%%
%%% Journal of Chemical Physics (2023) %%%

function [r_tilde,gr_tilde] = KAMEL_LOBE(r,RDF,varargin)
% INPUTS: 
% (1) r: vector of equispaced radii at which RDF is evaluated
% (2) RDF: vector of corresponding RDF values
% (3, optional): width of Gaussian kernel (set to 0.015 by default)
%
% OUTPUTS: 
% (1) r_tilde: vector of equispaced radii at which KAMEL-LOBE RDF is evaluated
% (2) gr_tilde: vector of corresponding KAMEL-LOBE RDF values

    if nargin < 3
        w  =  0.015;
    else
        w = varargin{1};
    end

    Nbin = size(RDF,1); % total number of bins
    delr = r(3,1)-r(2,1); % length of bin 
    m_KL = ceil(2*w/delr);
    
    if m_KL >= 2
    
        M1 = eye(Nbin) + tril(2*ones(Nbin),-1);
        M1(:,1) = 1;
        M1(1,:) = 0;
        M2 = diag((0:Nbin-1).^2).*delr.^2;
        T1 = 2.*pi.*delr.*M1*M2;
        
        
        k_KL = 2*m_KL-1;
        T2 = eye(Nbin,Nbin);
        fractions = zeros(1,k_KL);
        
        A1_block = eye(m_KL);
        A2_block = zeros(m_KL,Nbin-m_KL);
        B_block = T2(m_KL+1:end-m_KL,:);
        C1_block = A2_block;
        C2_block = A1_block;
        fractions(1,m_KL:end) = normcdf(((0:m_KL-1)+0.5)*delr,0,w)-normcdf(((0:m_KL-1)-0.5)*delr,0,w);             
        fractions(1,1:m_KL-1) = flip(fractions(1,m_KL+1:2*m_KL-1));
        fractions(1,:) = fractions(1,:)*(1/sum(fractions));   
        B_block = spdiags(repmat(fractions,Nbin-2*m_KL,1),0:2*(m_KL-1), Nbin-2*m_KL, Nbin);
        T2 = full([A1_block A2_block;B_block;C1_block C2_block]);
        sq_r = zeros(Nbin,1);
        sq_r(2:end,1) = ((1:Nbin-1).*delr).^-2;
        r_exploded_out = repmat(sq_r,[1 Nbin]);
        K = (-1).^((1:Nbin)+1) ;
        Ksq = K'*K;
        KTr = 2.*tril(Ksq);
        KTr(1:1+size(KTr,1):end) = 1;
        Kdiff = (2/delr).*KTr;
        T3 = (1/(4.*pi)).*r_exploded_out.*Kdiff;
        gr_convert = T3*T2*T1*RDF;
        gr_tilde = gr_convert(1:end-2*m_KL,1);    
        r_tilde = r(1:end-2*m_KL,1);
    
    else 
        gr_tilde = RDF; 
        r_tilde = r;  
        disp(' ##### NOTE: KAMEL-LOBE not performed as w <= delr/2 #####');
      
    end

end
    