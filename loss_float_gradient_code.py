import torch
def get_loss_float_gradient(pred_output_float,gt_output_float,gt_output_float_mask):
    return torch.sum( ( (pred_output_float[:,24,:-1,:-1,:-1]-pred_output_float[:,43,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,24,:-1,:-1,:-1],gt_output_float_mask[:,43,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,24,:-1,:-1,:-1]-gt_output_float[:,43,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,24,:-1,:-1,:-1]-pred_output_float[:,27,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,24,:-1,:-1,:-1],gt_output_float_mask[:,27,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,24,:-1,:-1,:-1]-gt_output_float[:,27,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,27,:-1,:-1,:-1]-pred_output_float[:,0,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,27,:-1,:-1,:-1],gt_output_float_mask[:,0,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,27,:-1,:-1,:-1]-gt_output_float[:,0,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,27,:-1,:-1,:-1]-pred_output_float[:,33,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,27,:-1,:-1,:-1],gt_output_float_mask[:,33,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,27,:-1,:-1,:-1]-gt_output_float[:,33,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,33,:-1,:-1,:-1]-pred_output_float[:,9,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,33,:-1,:-1,:-1],gt_output_float_mask[:,9,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,33,:-1,:-1,:-1]-gt_output_float[:,9,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,33,:-1,:-1,:-1]-pred_output_float[:,24,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,33,:-1,:-1,:-1],gt_output_float_mask[:,24,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,33,:-1,:-1,:-1]-gt_output_float[:,24,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,43,:-1,:-1,:-1]-pred_output_float[:,49,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,43,:-1,:-1,:-1],gt_output_float_mask[:,49,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,43,:-1,:-1,:-1]-gt_output_float[:,49,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,43,:-1,:-1,:-1]-pred_output_float[:,0,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,43,:-1,:-1,:-1],gt_output_float_mask[:,0,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,43,:-1,:-1,:-1]-gt_output_float[:,0,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,0,:-1,:-1,:-1]-pred_output_float[:,12,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,0,:-1,:-1,:-1],gt_output_float_mask[:,12,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,0,:-1,:-1,:-1]-gt_output_float[:,12,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,0,:-1,:-1,:-1]-pred_output_float[:,9,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,0,:-1,:-1,:-1],gt_output_float_mask[:,9,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,0,:-1,:-1,:-1]-gt_output_float[:,9,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,9,:-1,:-1,:-1]-pred_output_float[:,21,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,9,:-1,:-1,:-1],gt_output_float_mask[:,21,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,9,:-1,:-1,:-1]-gt_output_float[:,21,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,9,:-1,:-1,:-1]-pred_output_float[:,43,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,9,:-1,:-1,:-1],gt_output_float_mask[:,43,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,9,:-1,:-1,:-1]-gt_output_float[:,43,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,49,:-1,:-1,:-1]-pred_output_float[:,24,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,49,:-1,:-1,:-1],gt_output_float_mask[:,24,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,49,:-1,:-1,:-1]-gt_output_float[:,24,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,49,:-1,:-1,:-1]-pred_output_float[:,12,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,49,:-1,:-1,:-1],gt_output_float_mask[:,12,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,49,:-1,:-1,:-1]-gt_output_float[:,12,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,12,:-1,:-1,:-1]-pred_output_float[:,27,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,12,:-1,:-1,:-1],gt_output_float_mask[:,27,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,12,:-1,:-1,:-1]-gt_output_float[:,27,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,12,:-1,:-1,:-1]-pred_output_float[:,21,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,12,:-1,:-1,:-1],gt_output_float_mask[:,21,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,12,:-1,:-1,:-1]-gt_output_float[:,21,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,21,:-1,:-1,:-1]-pred_output_float[:,33,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,21,:-1,:-1,:-1],gt_output_float_mask[:,33,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,21,:-1,:-1,:-1]-gt_output_float[:,33,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,21,:-1,:-1,:-1]-pred_output_float[:,49,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,21,:-1,:-1,:-1],gt_output_float_mask[:,49,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,21,:-1,:-1,:-1]-gt_output_float[:,49,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,24,:-1,:-1,:-1]-pred_output_float[:,45,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,24,:-1,:-1,:-1],gt_output_float_mask[:,45,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,24,:-1,:-1,:-1]-gt_output_float[:,45,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,24,:-1,:-1,:-1]-pred_output_float[:,29,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,24,:-1,:-1,:-1],gt_output_float_mask[:,29,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,24,:-1,:-1,:-1]-gt_output_float[:,29,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,29,:-1,:-1,:-1]-pred_output_float[:,3,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,29,:-1,:-1,:-1],gt_output_float_mask[:,3,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,29,:-1,:-1,:-1]-gt_output_float[:,3,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,29,:-1,:-1,:-1]-pred_output_float[:,31,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,29,:-1,:-1,:-1],gt_output_float_mask[:,31,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,29,:-1,:-1,:-1]-gt_output_float[:,31,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,31,:-1,:-1,:-1]-pred_output_float[:,6,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,31,:-1,:-1,:-1],gt_output_float_mask[:,6,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,31,:-1,:-1,:-1]-gt_output_float[:,6,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,31,:-1,:-1,:-1]-pred_output_float[:,24,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,31,:-1,:-1,:-1],gt_output_float_mask[:,24,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,31,:-1,:-1,:-1]-gt_output_float[:,24,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,45,:-1,:-1,:-1]-pred_output_float[:,47,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,45,:-1,:-1,:-1],gt_output_float_mask[:,47,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,45,:-1,:-1,:-1]-gt_output_float[:,47,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,45,:-1,:-1,:-1]-pred_output_float[:,3,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,45,:-1,:-1,:-1],gt_output_float_mask[:,3,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,45,:-1,:-1,:-1]-gt_output_float[:,3,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,3,:-1,:-1,:-1]-pred_output_float[:,15,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,3,:-1,:-1,:-1],gt_output_float_mask[:,15,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,3,:-1,:-1,:-1]-gt_output_float[:,15,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,3,:-1,:-1,:-1]-pred_output_float[:,6,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,3,:-1,:-1,:-1],gt_output_float_mask[:,6,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,3,:-1,:-1,:-1]-gt_output_float[:,6,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,6,:-1,:-1,:-1]-pred_output_float[:,18,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,6,:-1,:-1,:-1],gt_output_float_mask[:,18,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,6,:-1,:-1,:-1]-gt_output_float[:,18,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,6,:-1,:-1,:-1]-pred_output_float[:,45,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,6,:-1,:-1,:-1],gt_output_float_mask[:,45,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,6,:-1,:-1,:-1]-gt_output_float[:,45,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,47,:-1,:-1,:-1]-pred_output_float[:,24,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,47,:-1,:-1,:-1],gt_output_float_mask[:,24,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,47,:-1,:-1,:-1]-gt_output_float[:,24,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,47,:-1,:-1,:-1]-pred_output_float[:,15,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,47,:-1,:-1,:-1],gt_output_float_mask[:,15,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,47,:-1,:-1,:-1]-gt_output_float[:,15,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,15,:-1,:-1,:-1]-pred_output_float[:,29,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,15,:-1,:-1,:-1],gt_output_float_mask[:,29,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,15,:-1,:-1,:-1]-gt_output_float[:,29,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,15,:-1,:-1,:-1]-pred_output_float[:,18,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,15,:-1,:-1,:-1],gt_output_float_mask[:,18,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,15,:-1,:-1,:-1]-gt_output_float[:,18,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,18,:-1,:-1,:-1]-pred_output_float[:,31,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,18,:-1,:-1,:-1],gt_output_float_mask[:,31,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,18,:-1,:-1,:-1]-gt_output_float[:,31,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,18,:-1,:-1,:-1]-pred_output_float[:,47,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,18,:-1,:-1,:-1],gt_output_float_mask[:,47,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,18,:-1,:-1,:-1]-gt_output_float[:,47,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,25,:-1,:-1,:-1]-pred_output_float[:,35,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,25,:-1,:-1,:-1],gt_output_float_mask[:,35,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,25,:-1,:-1,:-1]-gt_output_float[:,35,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,25,:-1,:-1,:-1]-pred_output_float[:,28,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,25,:-1,:-1,:-1],gt_output_float_mask[:,28,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,25,:-1,:-1,:-1]-gt_output_float[:,28,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,28,:-1,:-1,:-1]-pred_output_float[:,1,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,28,:-1,:-1,:-1],gt_output_float_mask[:,1,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,28,:-1,:-1,:-1]-gt_output_float[:,1,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,28,:-1,:-1,:-1]-pred_output_float[:,30,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,28,:-1,:-1,:-1],gt_output_float_mask[:,30,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,28,:-1,:-1,:-1]-gt_output_float[:,30,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,30,:-1,:-1,:-1]-pred_output_float[:,4,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,30,:-1,:-1,:-1],gt_output_float_mask[:,4,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,30,:-1,:-1,:-1]-gt_output_float[:,4,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,30,:-1,:-1,:-1]-pred_output_float[:,25,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,30,:-1,:-1,:-1],gt_output_float_mask[:,25,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,30,:-1,:-1,:-1]-gt_output_float[:,25,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,35,:-1,:-1,:-1]-pred_output_float[:,41,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,35,:-1,:-1,:-1],gt_output_float_mask[:,41,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,35,:-1,:-1,:-1]-gt_output_float[:,41,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,35,:-1,:-1,:-1]-pred_output_float[:,1,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,35,:-1,:-1,:-1],gt_output_float_mask[:,1,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,35,:-1,:-1,:-1]-gt_output_float[:,1,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,1,:-1,:-1,:-1]-pred_output_float[:,13,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,1,:-1,:-1,:-1],gt_output_float_mask[:,13,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,1,:-1,:-1,:-1]-gt_output_float[:,13,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,1,:-1,:-1,:-1]-pred_output_float[:,4,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,1,:-1,:-1,:-1],gt_output_float_mask[:,4,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,1,:-1,:-1,:-1]-gt_output_float[:,4,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,4,:-1,:-1,:-1]-pred_output_float[:,16,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,4,:-1,:-1,:-1],gt_output_float_mask[:,16,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,4,:-1,:-1,:-1]-gt_output_float[:,16,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,4,:-1,:-1,:-1]-pred_output_float[:,35,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,4,:-1,:-1,:-1],gt_output_float_mask[:,35,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,4,:-1,:-1,:-1]-gt_output_float[:,35,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,41,:-1,:-1,:-1]-pred_output_float[:,25,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,41,:-1,:-1,:-1],gt_output_float_mask[:,25,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,41,:-1,:-1,:-1]-gt_output_float[:,25,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,41,:-1,:-1,:-1]-pred_output_float[:,13,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,41,:-1,:-1,:-1],gt_output_float_mask[:,13,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,41,:-1,:-1,:-1]-gt_output_float[:,13,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,13,:-1,:-1,:-1]-pred_output_float[:,28,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,13,:-1,:-1,:-1],gt_output_float_mask[:,28,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,13,:-1,:-1,:-1]-gt_output_float[:,28,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,13,:-1,:-1,:-1]-pred_output_float[:,16,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,13,:-1,:-1,:-1],gt_output_float_mask[:,16,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,13,:-1,:-1,:-1]-gt_output_float[:,16,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,16,:-1,:-1,:-1]-pred_output_float[:,30,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,16,:-1,:-1,:-1],gt_output_float_mask[:,30,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,16,:-1,:-1,:-1]-gt_output_float[:,30,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,16,:-1,:-1,:-1]-pred_output_float[:,41,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,16,:-1,:-1,:-1],gt_output_float_mask[:,41,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,16,:-1,:-1,:-1]-gt_output_float[:,41,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,25,:-1,:-1,:-1]-pred_output_float[:,37,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,25,:-1,:-1,:-1],gt_output_float_mask[:,37,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,25,:-1,:-1,:-1]-gt_output_float[:,37,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,25,:-1,:-1,:-1]-pred_output_float[:,34,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,25,:-1,:-1,:-1],gt_output_float_mask[:,34,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,25,:-1,:-1,:-1]-gt_output_float[:,34,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,34,:-1,:-1,:-1]-pred_output_float[:,10,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,34,:-1,:-1,:-1],gt_output_float_mask[:,10,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,34,:-1,:-1,:-1]-gt_output_float[:,10,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,34,:-1,:-1,:-1]-pred_output_float[:,32,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,34,:-1,:-1,:-1],gt_output_float_mask[:,32,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,34,:-1,:-1,:-1]-gt_output_float[:,32,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,32,:-1,:-1,:-1]-pred_output_float[:,7,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,32,:-1,:-1,:-1],gt_output_float_mask[:,7,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,32,:-1,:-1,:-1]-gt_output_float[:,7,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,32,:-1,:-1,:-1]-pred_output_float[:,25,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,32,:-1,:-1,:-1],gt_output_float_mask[:,25,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,32,:-1,:-1,:-1]-gt_output_float[:,25,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,37,:-1,:-1,:-1]-pred_output_float[:,39,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,37,:-1,:-1,:-1],gt_output_float_mask[:,39,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,37,:-1,:-1,:-1]-gt_output_float[:,39,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,37,:-1,:-1,:-1]-pred_output_float[:,10,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,37,:-1,:-1,:-1],gt_output_float_mask[:,10,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,37,:-1,:-1,:-1]-gt_output_float[:,10,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,10,:-1,:-1,:-1]-pred_output_float[:,22,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,10,:-1,:-1,:-1],gt_output_float_mask[:,22,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,10,:-1,:-1,:-1]-gt_output_float[:,22,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,10,:-1,:-1,:-1]-pred_output_float[:,7,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,10,:-1,:-1,:-1],gt_output_float_mask[:,7,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,10,:-1,:-1,:-1]-gt_output_float[:,7,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,7,:-1,:-1,:-1]-pred_output_float[:,19,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,7,:-1,:-1,:-1],gt_output_float_mask[:,19,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,7,:-1,:-1,:-1]-gt_output_float[:,19,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,7,:-1,:-1,:-1]-pred_output_float[:,37,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,7,:-1,:-1,:-1],gt_output_float_mask[:,37,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,7,:-1,:-1,:-1]-gt_output_float[:,37,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,39,:-1,:-1,:-1]-pred_output_float[:,25,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,39,:-1,:-1,:-1],gt_output_float_mask[:,25,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,39,:-1,:-1,:-1]-gt_output_float[:,25,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,39,:-1,:-1,:-1]-pred_output_float[:,22,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,39,:-1,:-1,:-1],gt_output_float_mask[:,22,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,39,:-1,:-1,:-1]-gt_output_float[:,22,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,22,:-1,:-1,:-1]-pred_output_float[:,34,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,22,:-1,:-1,:-1],gt_output_float_mask[:,34,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,22,:-1,:-1,:-1]-gt_output_float[:,34,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,22,:-1,:-1,:-1]-pred_output_float[:,19,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,22,:-1,:-1,:-1],gt_output_float_mask[:,19,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,22,:-1,:-1,:-1]-gt_output_float[:,19,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,19,:-1,:-1,:-1]-pred_output_float[:,32,:-1,:-1,1:])**2 )*torch.min(gt_output_float_mask[:,19,:-1,:-1,:-1],gt_output_float_mask[:,32,:-1,:-1,1:] )*( torch.abs(gt_output_float[:,19,:-1,:-1,:-1]-gt_output_float[:,32,:-1,:-1,1:])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,19,:-1,:-1,:-1]-pred_output_float[:,39,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,19,:-1,:-1,:-1],gt_output_float_mask[:,39,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,19,:-1,:-1,:-1]-gt_output_float[:,39,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,26,:-1,:-1,:-1]-pred_output_float[:,36,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,26,:-1,:-1,:-1],gt_output_float_mask[:,36,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,26,:-1,:-1,:-1]-gt_output_float[:,36,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,26,:-1,:-1,:-1]-pred_output_float[:,44,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,26,:-1,:-1,:-1],gt_output_float_mask[:,44,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,26,:-1,:-1,:-1]-gt_output_float[:,44,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,44,:-1,:-1,:-1]-pred_output_float[:,2,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,44,:-1,:-1,:-1],gt_output_float_mask[:,2,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,44,:-1,:-1,:-1]-gt_output_float[:,2,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,44,:-1,:-1,:-1]-pred_output_float[:,46,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,44,:-1,:-1,:-1],gt_output_float_mask[:,46,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,44,:-1,:-1,:-1]-gt_output_float[:,46,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,46,:-1,:-1,:-1]-pred_output_float[:,5,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,46,:-1,:-1,:-1],gt_output_float_mask[:,5,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,46,:-1,:-1,:-1]-gt_output_float[:,5,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,46,:-1,:-1,:-1]-pred_output_float[:,26,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,46,:-1,:-1,:-1],gt_output_float_mask[:,26,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,46,:-1,:-1,:-1]-gt_output_float[:,26,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,36,:-1,:-1,:-1]-pred_output_float[:,38,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,36,:-1,:-1,:-1],gt_output_float_mask[:,38,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,36,:-1,:-1,:-1]-gt_output_float[:,38,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,36,:-1,:-1,:-1]-pred_output_float[:,2,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,36,:-1,:-1,:-1],gt_output_float_mask[:,2,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,36,:-1,:-1,:-1]-gt_output_float[:,2,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,2,:-1,:-1,:-1]-pred_output_float[:,11,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,2,:-1,:-1,:-1],gt_output_float_mask[:,11,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,2,:-1,:-1,:-1]-gt_output_float[:,11,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,2,:-1,:-1,:-1]-pred_output_float[:,5,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,2,:-1,:-1,:-1],gt_output_float_mask[:,5,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,2,:-1,:-1,:-1]-gt_output_float[:,5,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,5,:-1,:-1,:-1]-pred_output_float[:,8,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,5,:-1,:-1,:-1],gt_output_float_mask[:,8,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,5,:-1,:-1,:-1]-gt_output_float[:,8,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,5,:-1,:-1,:-1]-pred_output_float[:,36,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,5,:-1,:-1,:-1],gt_output_float_mask[:,36,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,5,:-1,:-1,:-1]-gt_output_float[:,36,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,38,:-1,:-1,:-1]-pred_output_float[:,26,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,38,:-1,:-1,:-1],gt_output_float_mask[:,26,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,38,:-1,:-1,:-1]-gt_output_float[:,26,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,38,:-1,:-1,:-1]-pred_output_float[:,11,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,38,:-1,:-1,:-1],gt_output_float_mask[:,11,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,38,:-1,:-1,:-1]-gt_output_float[:,11,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,11,:-1,:-1,:-1]-pred_output_float[:,44,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,11,:-1,:-1,:-1],gt_output_float_mask[:,44,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,11,:-1,:-1,:-1]-gt_output_float[:,44,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,11,:-1,:-1,:-1]-pred_output_float[:,8,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,11,:-1,:-1,:-1],gt_output_float_mask[:,8,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,11,:-1,:-1,:-1]-gt_output_float[:,8,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,8,:-1,:-1,:-1]-pred_output_float[:,46,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,8,:-1,:-1,:-1],gt_output_float_mask[:,46,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,8,:-1,:-1,:-1]-gt_output_float[:,46,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,8,:-1,:-1,:-1]-pred_output_float[:,38,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,8,:-1,:-1,:-1],gt_output_float_mask[:,38,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,8,:-1,:-1,:-1]-gt_output_float[:,38,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,26,:-1,:-1,:-1]-pred_output_float[:,42,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,26,:-1,:-1,:-1],gt_output_float_mask[:,42,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,26,:-1,:-1,:-1]-gt_output_float[:,42,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,26,:-1,:-1,:-1]-pred_output_float[:,50,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,26,:-1,:-1,:-1],gt_output_float_mask[:,50,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,26,:-1,:-1,:-1]-gt_output_float[:,50,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,50,:-1,:-1,:-1]-pred_output_float[:,14,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,50,:-1,:-1,:-1],gt_output_float_mask[:,14,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,50,:-1,:-1,:-1]-gt_output_float[:,14,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,50,:-1,:-1,:-1]-pred_output_float[:,48,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,50,:-1,:-1,:-1],gt_output_float_mask[:,48,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,50,:-1,:-1,:-1]-gt_output_float[:,48,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,48,:-1,:-1,:-1]-pred_output_float[:,17,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,48,:-1,:-1,:-1],gt_output_float_mask[:,17,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,48,:-1,:-1,:-1]-gt_output_float[:,17,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,48,:-1,:-1,:-1]-pred_output_float[:,26,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,48,:-1,:-1,:-1],gt_output_float_mask[:,26,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,48,:-1,:-1,:-1]-gt_output_float[:,26,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,42,:-1,:-1,:-1]-pred_output_float[:,40,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,42,:-1,:-1,:-1],gt_output_float_mask[:,40,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,42,:-1,:-1,:-1]-gt_output_float[:,40,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,42,:-1,:-1,:-1]-pred_output_float[:,14,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,42,:-1,:-1,:-1],gt_output_float_mask[:,14,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,42,:-1,:-1,:-1]-gt_output_float[:,14,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,14,:-1,:-1,:-1]-pred_output_float[:,23,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,14,:-1,:-1,:-1],gt_output_float_mask[:,23,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,14,:-1,:-1,:-1]-gt_output_float[:,23,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,14,:-1,:-1,:-1]-pred_output_float[:,17,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,14,:-1,:-1,:-1],gt_output_float_mask[:,17,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,14,:-1,:-1,:-1]-gt_output_float[:,17,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,17,:-1,:-1,:-1]-pred_output_float[:,20,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,17,:-1,:-1,:-1],gt_output_float_mask[:,20,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,17,:-1,:-1,:-1]-gt_output_float[:,20,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,17,:-1,:-1,:-1]-pred_output_float[:,42,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,17,:-1,:-1,:-1],gt_output_float_mask[:,42,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,17,:-1,:-1,:-1]-gt_output_float[:,42,1:,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,40,:-1,:-1,:-1]-pred_output_float[:,26,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,40,:-1,:-1,:-1],gt_output_float_mask[:,26,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,40,:-1,:-1,:-1]-gt_output_float[:,26,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,40,:-1,:-1,:-1]-pred_output_float[:,23,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,40,:-1,:-1,:-1],gt_output_float_mask[:,23,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,40,:-1,:-1,:-1]-gt_output_float[:,23,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,23,:-1,:-1,:-1]-pred_output_float[:,50,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,23,:-1,:-1,:-1],gt_output_float_mask[:,50,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,23,:-1,:-1,:-1]-gt_output_float[:,50,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,23,:-1,:-1,:-1]-pred_output_float[:,20,:-1,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,23,:-1,:-1,:-1],gt_output_float_mask[:,20,:-1,:-1,:-1] )*( torch.abs(gt_output_float[:,23,:-1,:-1,:-1]-gt_output_float[:,20,:-1,:-1,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,20,:-1,:-1,:-1]-pred_output_float[:,48,:-1,1:,:-1])**2 )*torch.min(gt_output_float_mask[:,20,:-1,:-1,:-1],gt_output_float_mask[:,48,:-1,1:,:-1] )*( torch.abs(gt_output_float[:,20,:-1,:-1,:-1]-gt_output_float[:,48,:-1,1:,:-1])<2e-4 ).float() ) + torch.sum( ( (pred_output_float[:,20,:-1,:-1,:-1]-pred_output_float[:,40,1:,:-1,:-1])**2 )*torch.min(gt_output_float_mask[:,20,:-1,:-1,:-1],gt_output_float_mask[:,40,1:,:-1,:-1] )*( torch.abs(gt_output_float[:,20,:-1,:-1,:-1]-gt_output_float[:,40,1:,:-1,:-1])<2e-4 ).float() )