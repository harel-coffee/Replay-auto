function varargout = corrSeqSpeed(varargin)
%
% [t,seqSpeed] = corrSeqSpeed(timeStamps,typedSeq,targSeq[,tSpan,dt,interpMeth,circShift])
%
% Required Inputs:
%   timeStamps - numeric vector of length M with keypress timestamps in ms
%   typedSeq - numeric or string vector of length M with keypress IDs (i.e.- '42314423144231442314423')
%   targSeq - numeric or string vector of length N with target sequence (i.e. - '42314')
%
% Optional Inputs
%   tSpan - bounding limits for t in ms (i.e. - [1 10000]). DEFAULT is [0 timeStamp(end)]
%   dt - delta t in ms (determines sampling rate of output; DEFAULT = 1ms)
%   interpMeth - interpolation method (type "help interp1" for all options. DEFAULT is 'makima' cubic hermite spline interpolation)
%   circShift - BOOLEAN switch.  DEFAULT is FALSE (TRUE looks for all circular shifts of targSeq in typedSeq, improving resolution of estimate). 
%               Not recommended if long pauses between sequence repetitions are present in data.
%
% Outputs
%   t - time vector
%   seqSpeed - vector of correct sequence speed
%
%--------------------------------------------------------------------------
% v1.0 21 August 2019 by Ethan R Buch
%
% v1.1 23 August 2019 by Ethan R Buch
%   Changed extrapolation behavior.  Instead of appending NaNs to all samples before first and after last completed correct sequences,
%   the speed value for first complete correct sequence is replicated across all prior samples, and the speed value for last complete  
%   correct sequence is replicated across all subsequent samples.
%
% v1.2 4 September 2019 by Ethan R Buch
%   Added check to return NaNs if no correct sequences found
%
% v1.3 16 July 2021 by Ethan R Buch
%   Improved measure of instantaneous correct sequence speed. Removed
%   cyclical bias caused by different mix of transitions for different circular
%   shifts

%% Input parsing and I/O checks
curVer = '1.3';
disp(['Running ' mfilename('fullpath') ' ' curVer]);

if nargin < 3
    error('A minimum of three inputs (timeStamps, typedSeq and targSeq) are required.');
end
if nargin >= 3
    timeStamps = varargin{1};
    if ~isnumeric(timeStamps)
        error('timeStamps input must be a numeric array.');
    end
    typedSeq = varargin{2};
    if isnumeric(typedSeq) %Convert numeric array to string
        typedSeq = strrep(num2str(typedSeq),' ','');
    end
    targSeq = varargin{3};
    if isnumeric(targSeq) %Convert numeric array to string
        targSeq = strrep(num2str(targSeq),' ','');
    end
end
if length(timeStamps) ~= length(typedSeq)
    error('timeStamps and typedSeq input vectors must be the same length.');
end
    
if nargin > 3
    tSpan = varargin{4};
else
    tSpan = [1 timeStamps(end)];
end
if nargin > 4
    dt = varargin{5};
else
    dt = 1; %1000Hz if timeStamps are in ms
end
if nargin > 5
    interpMeth = varargin{6};
else
    interpMeth = 'makima';
end
if nargin > 6
    circShift = varargin{7};
else
    circShift = false;
end

if nargout~=2
    error('Two outputs (t and seqSpeed) required.');
end

%% Compute time-resolved correct sequence speed estimate
%Store typed and target sequence lengths
M = length(typedSeq);
N = length(targSeq);
speedMat = NaN(N,M);
if circShift
    allShifts = 0:N-1;
else
    allShifts = 0;
end
anyCorrSeqs = false;
for jS = allShifts %Loop through all circular shifts of target sequence
    iSeqStart = regexp(typedSeq,circshift(targSeq,-jS));
    if ~isempty(iSeqStart)
        anyCorrSeqs = true;
    end
    for curSS = iSeqStart %Loop through each correct sequence onset index
        speedMat(jS+1,curSS:curSS+4) = (timeStamps(curSS+4) - timeStamps(curSS)).^-1.*1e3; %Replicate speed estimate over full length of correct sequence
    end
end

%%% Mean correct for all shifted versions of sequence and then center on
%%% mean for 0-shift version (i.e. - 41324).  This removes speed bias
%%% caused by different mix of transitions. Could also scale variance via z-transformation but
%%% not as robust for early trials with more errors (i.e. - only 1 matching sequence means zero variance).
speedMat_bc = speedMat - ...
    repmat(nanmean(speedMat,2),1,size(speedMat,2)) ... %Zero-mean for all shifts
    + repmat(nanmean(speedMat(1,:)),size(speedMat)); ... %Center on mean for zero-shift sequence (i.e. - 41324)

mnSpeed = nanmean(speedMat_bc,1); %Calculate mean speed over all shifts for each keypress timepoint
iObs = isfinite(mnSpeed) & isfinite(timeStamps); %Index finite speed values for interpolation step
t = tSpan(1):dt:tSpan(end); %Generate time vector for interpolation step
if sum(iObs)==0
    warning('Unable to calculate speed values. Returning all NaNs for sequence speed output.')
    seqSpeed = NaN(size(t));
end
if anyCorrSeqs && sum(iObs)>0
    seqSpeed = interp1(timeStamps(iObs),mnSpeed(iObs),t,interpMeth,NaN); %Interpolate over full timespan using method "interMeth" and setting all extrapulated points to NaN
    seqSpeed(t<timeStamps(find(iObs,1,'first'))) = mnSpeed(find(iObs,1,'first')); %Replicate first non-NaN value onto earlier time-points
    seqSpeed(t>timeStamps(find(iObs,1,'last'))) = mnSpeed(find(iObs,1,'last')); %Replicate last non-NaN value onto later time-points
else
    warning('No correct sequences found. Returning all NaNs for sequence speed output.'); %Throw a warning if no correct sequences are found
    seqSpeed = NaN(size(t));
end

%% Set outputs
varargout(1) = {t};
varargout(2) = {seqSpeed};
