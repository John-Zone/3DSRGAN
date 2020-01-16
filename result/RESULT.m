load result
HR = squeeze(HR);
LR = squeeze(LR);
OR = squeeze(ori);
enviwrite(HR,160,160,191,'hr');
enviwrite(LR,160,160,191,'LR');
enviwrite(OR,40,40,191,'OR');