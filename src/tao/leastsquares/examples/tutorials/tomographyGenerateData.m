% PetscBinaryWrite, Default: Big Endian, 'indices','int32'
% (1) Generate Object
% sampleName: the type and name to specify a sample object, not case-sensitive
% Ny*Nx:    Sample object size
% WGT:      Sample object ground truth
sampleName = 'Phantom'; % choose from {'Phantom', 'Brain', ''Golosio', 'circle', 'checkboard', 'fakeRod'}; not case-sensitive
Nx = 50; Ny = Nx; WSz = [Ny, Nx]; %  XH: Ny = 100 -> 10 or 50 for debug.  currently assume object shape is square not general rectangle
WGT = createObject(sampleName, WSz);

% (2) Scan Object: Create Forward Model and Sinograms
% NTheta*NTau:  Sinogram Size
% L:            Forward Model, sparse matrix of NTheta*NTau by Ny*Nx
% SAll:         Sinogram for all scans of NTheta*NTau*NScan, SAll(:,:,n) for the nth scan
NTheta = 20;  % sample angle #. Use odd NOT even, for display purpose of sinagram of Phantom. As even NTheta will include theta of 90 degree where sinagram will be very bright as Phantom sample has verical bright line on left and right boundary.
NTau = ceil(sqrt(sum(WSz.^2))); NTau = NTau + rem(NTau-Ny,2); % number of discrete beam, enough to cover object diagonal, also use + rem(NTau-Ny,2) to make NTau the same odd/even as Ny just for perfection, so that for theta=0, we have sum(WGT, 2)' and  S(1, (1:Ny)+(NTau-Ny)/2) are the same with a scale factor
SSz = [NTheta, NTau];
L = XTM_Tensor_XH(WSz, NTheta, NTau);
S = reshape(L*WGT(:), NTheta, NTau);

%% Save data in petsc binary format, b = A*x
% save to one file
PetscBinaryWrite('tomographyData_A_b_xGT', L, S(:), WGT(:),'precision', 'float64');
[A2, b2, xGT2] = PetscBinaryRead('tomographyData_A_b_xGT');
difference(full(A2), full(L));
difference(b2, S(:));
difference(xGT2, WGT(:));
% Save to separate files
% PetscBinaryWrite('tomographySparseMatrixA', L, 'precision', 'float64');
% PetscBinaryWrite('tomographyVecXGT', WGT(:), 'precision', 'float64');
% PetscBinaryWrite('tomographyVecB', S(:), 'precision', 'float64');

%% Compare Groundtruth vs BRGN reconstruction vs (optional TwIST result)
% Ground truth and model
[A, b, xGT] = PetscBinaryRead('tomographyData_A_b_xGT');
Nx = sqrt(numel(xGT)); Ny = Nx; WSz = [Ny, Nx];
WGT = reshape(xGT, WSz);
% petsc reconstruction
xRecBRGN = PetscBinaryRead('tomographyResult_x');    
WRecBRGN = reshape(xRecBRGN, WSz);
% Prepare for figure
WAll = {WGT, WRecBRGN};
titleAll = {'Ground Truth', sprintf('Reconstruction-Tao-brgn-nonnegative,psnr=%.2fdB', psnr(WRecBRGN, WGT))};
  
% May add the Matlab reconstruction using TwIST to comparison
isDemoMatLabReconstruction = 1; % 1/0
if isDemoMatLabReconstruction  
    % Reconstruction with solver from XH, with L1/TV regularizer.
    % Need 100/500/1000+ iteration to get good/very good/excellent result with small regularizer.
    % choose small maxSVDofA to make sure initial step size is not too small. 1.8e-6 and 1e-6 could make big difference for n=2 case
    regType = 'L1'; % 'TV' or 'L1' % TV is better and cleaner for phantom example        
    regWt = 1e-8*max(WGT(:)); % 1e-6 to 1e-8 both good for phantom, %1e-8*max(WGT(:)) use 1e-8 for brain, especically WGT is scaled to maximum of 1 not 40
    maxIterA = 500; % 100 is not enough? 500 is
    maxSVDofA = 1e-6; %svds(A, 1)*1e-4; % multiply by 1e-4 to make sure it is small enough so that first step in TwIST is long enough
    paraTwist = {'xSz', WSz, 'regFun', regType, 'regWt', regWt, 'isNonNegative', 1, 'maxIterA', maxIterA, 'xGT', xGT, 'maxSVDofA', maxSVDofA, 'tolA', 1e-8};
    xRecTwist = solveTwist(b, A, paraTwist{:});
    WRecTwist = reshape(xRecTwist, WSz);
    WAll = [WAll, {WRecTwist}];
    titleAll = [titleAll, {sprintf('Reconstruction-Matlab-Twist, psnr=%.2fdB', psnr(WRecTwist, WGT))}];
end
%
% show results
figure(99); clf; multAxes(@imagesc, WAll); multAxes(@axis, 'image'); linkAxesXYZLimColorView; multAxes(@colorbar);
multAxes(@title, titleAll);

    
%% Generate data for cs1
testPetscBinaryWriteAndRead_cs1 = 1;
if testPetscBinaryWriteAndRead_cs1
    M=3; N=5;  rng(0); A = rand(M, N); A = round(A*100)/100; % generate A below
%     A = [0.81  0.91  0.28  0.96  0.96
%         0.91  0.63  0.55  0.16  0.49
%         0.13  0.10  0.96  0.97  0.80];
    xGT = [0;0;1;0;0];
    b = A*xGT; % [0.28; 0.55; 0.96];
    PetscBinaryWrite('cs1Data_A_b_xGT', A, b, xGT, 'precision', 'float64'); %   
    [A2, b2, xGT2] = PetscBinaryRead('cs1Data_A_b_xGT'); %cs1Data_A_b_xGT or jointsparsity1Data_A_b_xGT    
    % dictionary matrix, not used here
%     D = [-1 1 0 0 0;
%         0 -1 1 0 0;
%         0 0 -1 1 0;
%         0 0 0 -1 1];a
end
% xRecBRGN = PetscBinaryRead('jointsparsityResult_x');  norm(xRecBRGN-xGT)/norm(xGT) 

%% Generate data for jointsparsity1:  very small example
testPetscBinaryWriteAndRead_jointsparsity1 = 1;
if testPetscBinaryWriteAndRead_jointsparsity1
    M=3; N=5;  rng(0); A = rand(M, N); A = round(A*100)/100; % generate A below
%     A = [0.81  0.91  0.28  0.96  0.96
%         0.91  0.63  0.55  0.16  0.49
%         0.13  0.10  0.96  0.97  0.80];
    xGT =  [0;0;1;0;0]; %xGT = [xGT xGT]; %xGT = [xGT [0;0;0;1;0]];
    %xGT =  [[0;0;1;0;0] [0;0;0;1;0]];
    L = size(xGT,2);
    b = A*xGT; % [0.28; 0.55; 0.96];
    % in PetscBinaryWrite, the code "for l=1:nargin-1" will write 'precision' and 'float64' as two double vectors
    PetscBinaryWrite('jointsparsity1Data_A_b_xGT_L', A, b(:), xGT(:), L, 'precision', 'float64'); %   
    [A2, b2, xGT2, L2, dummy] = PetscBinaryRead('jointsparsity1Data_A_b_xGT_L'); %cs1Data_A_b_xGT or jointsparsity1Data_A_b_xGT    
    
    % dictionary matrix, not used here
%     D = [-1 1 0 0 0;
%         0 -1 1 0 0;
%         0 0 -1 1 0;
%         0 0 0 -1 1];
end
% xRecBRGN = PetscBinaryRead('jointsparsityResult_x');  norm(xRecBRGN-xGT)/norm(xGT) 

%% Generate data for jointsparsity1:  mideum size example
testPetscBinaryWriteAndRead_jointsparsity1 = 1;
if testPetscBinaryWriteAndRead_jointsparsity1
    M=50; N=100; L=4; rng(0); A = rand(M, N); A = round(A*100)/100; % generate A below
    rng(0); xGT =  rand(N,1); xGT(xGT<0.9)=0; 
    rng(0); xGT= bsxfun(@times, xGT, rand(1,L)); %ones(1,L);
    %xGT =  [[0;0;1;0;0] [0;0;0;1;0]];
    L = size(xGT,2);
    b = A*xGT; % [0.28; 0.55; 0.96];
    bNoisy = b + normrnd(0, 0.05*std(b(:)), size(b)); b = bNoisy;% add noise
    % in PetscBinaryWrite, the code "for l=1:nargin-1" will write 'precision' and 'float64' as two double vectors
    PetscBinaryWrite('jointsparsity1Data_A_b_xGT_L', A, b(:), xGT(:), L, 'precision', 'float64'); %   
    [A2, b2, xGT2, L2, dummy] = PetscBinaryRead('jointsparsity1Data_A_b_xGT_L'); %cs1Data_A_b_xGT or jointsparsity1Data_A_b_xGT    
    
    % dictionary matrix, not used here
%     D = [-1 1 0 0 0;
%         0 -1 1 0 0;
%         0 0 -1 1 0;
%         0 0 0 -1 1];
end
%%
xRecBRGN = PetscBinaryRead('jointsparsityResult_x');  norm(xRecBRGN-xGT(:))/norm(xGT(:));
figure, plot(xGT(:), 'r'); hold on; plot(xRecBRGN, 'b');

% make ./jointsparsity1; ./jointsparsity1 -tao_monitor -tao_max_it 100 -tao_brgn_regularization_type user -tao_brgn_regularizer_weight 1e-8 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-8
% relative reconstruction error: ||x-xGT||/||xGT|| = 1.5208e-02. L=1

%make ./jointsparsity1; ./jointsparsity1 -tao_monitor -tao_max_it 100 -tao_brgn_regularization_type l1dict -tao_brgn_regularizer_weight 1e-8 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-8
% relative reconstruction error: ||x-xGT||/||xGT|| = 6.9864e-01.  L=1
% relative reconstruction error: ||x-xGT||/||xGT|| = 7.0638e-01.  L=3