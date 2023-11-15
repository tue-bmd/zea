% usbmd_Globals.m  sets some globals for the USBMD Verasonics Tools
%                  [] means the default value

usbmd_g_TransName          = 'S5-1';          % 'ViewFlex' or 'CL15-7' or 'L15-7io' or 'S5-1'
usbmd_g_MatFilesDir        = 'C:\Users\Administrator\Documents\Vantage-4.6.2-2110271004\Harm\MatFiles';
usbmd_g_DataSaveDir        = 'C:\data\verasonics\2023-11-08';
usbmd_g_TransFreq          = [];
usbmd_g_TransNumElements   = [];
usbmd_g_TransVoltage       = 40;
usbmd_g_TGCSetPoints       = [0,400,400,400,400,400,400,400]; %[0,297,424,515,627,764,871,1000];

usbmd_g_SpeedOfSound       = 1540;

% Do not change the next line
usbmd_g_SupportedTxFreqMHz = [41.67 31.25 25.0 20.83 17.86 15.625 13.89 12.5, 10.4167, 8.9286, 7.8125, 6.944, 6.25];
