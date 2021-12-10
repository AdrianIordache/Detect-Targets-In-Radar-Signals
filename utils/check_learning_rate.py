import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir  = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import *

CFG = {
    'learning_rate' : 1e-6,
    'scheduler_name': 'OneCycleLR',

    'T_0': 62, # ['CosineAnnealingWarmRestarts', (59, 2, 12) (62, 3, 9)]
    'T_max': 10,
    'T_mult': 3,
    'min_lr': 1e-7,
    'max_lr': 3e-3,

    'no_batches': 2480, # 1231, # 1652,
    'batch_size': 5,

    'warmup_epochs': 1,
    'cosine_epochs': 8,
    'epochs': 9,

    'update_per_batch': True,
    'print_freq': 50
}

def get_scheduler(optimizer, scheduler_params = CFG):
    if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0              = scheduler_params['T_0'],
            T_mult           = scheduler_params['T_mult'],
            eta_min          = scheduler_params['min_lr'],
            last_epoch       = -1,
        )
    elif scheduler_params['scheduler_name'] == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer,
            max_lr          = scheduler_params['max_lr'],
            steps_per_epoch = scheduler_params['no_batches'],
            epochs          = scheduler_params['epochs'],
        )
    elif scheduler_params['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max          = scheduler_params['T_max'],
            eta_min        = scheduler_params['min_lr'],
            last_epoch     = -1
        )
    elif scheduler_params["scheduler_name"] == "GradualWarmupSchedulerV2 + CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, scheduler_params["cosine_epochs"])
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier = scheduler_params['T_mult'], total_epoch = scheduler_params["warmup_epochs"], after_scheduler = scheduler)
        return scheduler_warmup

    return scheduler


dataset = datasets.FakeData(size = CFG['no_batches'] * CFG['batch_size'], transform = transforms.ToTensor())

loader = DataLoader(
    dataset,
    batch_size   = CFG['batch_size'],
    shuffle      = False,
    num_workers  = 0, 
    drop_last    = True
)

model     = nn.Linear(3 * 224 * 224, 10)
optimizer = optim.SGD(model.parameters(), lr = CFG['learning_rate'])
scheduler = get_scheduler(optimizer, CFG)
criterion = nn.NLLLoss()

lrs = []
for epoch in range(CFG['epochs']):
    print(f"EPOCH: {epoch}")
    for step, (data, target) in enumerate(loader):
        if step % CFG['print_freq'] == 0 or step == (len(loader) - 1):
            print('[Epoch]: {}, [Batch]: {}, [LR]: {}'.format(
                epoch, step, np.round(scheduler.get_lr()[0], 6)))
            
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()

        output = model(data.view(CFG['batch_size'], -1))
        
        loss = criterion(output, target.long())
        loss.backward()

        optimizer.step()

        if CFG['update_per_batch'] == True: scheduler.step()

    if CFG['update_per_batch'] == False: scheduler.step()

xcoords = [CFG['no_batches'] * x for x in range(CFG['epochs'])]
plt.figure(figsize = (10, 18))
for xc in xcoords:
    plt.axvline(x = xc, color = 'red')
plt.plot(lrs)
plt.show()