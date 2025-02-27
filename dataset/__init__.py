import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *

from .atlas import Atlas
from .brat import Brat
from .ddti import DDTI
from .isic import ISIC2016
from .kits import KITS
from .lidc import LIDC
from .lnq import LNQ
from .pendal import Pendal
from .refuge import REFUGE
from .segrap import SegRap
from .stare import STARE
from .toothfairy import ToothFairy
from .wbc import WBC
from .multitask import Multitask
from .odoc import ODOC
from .relabel import Relabel
from .vessel import Vessel
from .od import OD
from .oc import OC
from .ex import EX
from .he import HE


def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])
    
    if args.dataset == 'isic':
        '''isic data'''
        isic_train_dataset = ISIC2016(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        isic_test_dataset = ISIC2016(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(isic_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(isic_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'Multitask':
        '''Multitask data'''
        multitask_train_dataset = Multitask(args, split='train')
        multitask_test_dataset = Multitask(args, split='val')

        nice_train_loader = DataLoader(multitask_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(multitask_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''
    
    elif args.dataset == 'relabel':
        '''relabel data'''
        relabel_train_dataset = Relabel(args, split='train')
        relabel_test_dataset = Relabel(args, split='test')

        nice_train_loader = DataLoader(relabel_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(relabel_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    
    elif args.dataset == 'odoc':
        '''odoc data'''
        odoc_train_dataset = ODOC(args, split='train')
        odoc_test_dataset = ODOC(args, split='val')

        nice_train_loader = DataLoader(odoc_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(odoc_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'vessel':
        '''vessel data'''
        vessel_train_dataset = Vessel(args, split='train')
        vessel_test_dataset = Vessel(args, split='val')

        nice_train_loader = DataLoader(vessel_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(vessel_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''
    
    elif args.dataset == 'od':
        '''od data'''
        od_train_dataset = OD(args, split='train')
        od_test_dataset = OD(args, split='val')

        nice_train_loader = DataLoader(od_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(od_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''
    
    elif args.dataset == 'oc':
        '''oc data'''
        oc_train_dataset = OC(args, split='train')
        oc_test_dataset = OC(args, split='val')

        nice_train_loader = DataLoader(oc_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(oc_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''
    
    elif args.dataset == 'ex':
        '''ex data'''
        ex_train_dataset = EX(args, split='train')
        ex_test_dataset = EX(args, split='val')

        nice_train_loader = DataLoader(ex_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(ex_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''
    
    elif args.dataset == 'he':
        '''he data'''
        he_train_dataset = HE(args, split='train')
        he_test_dataset = HE(args, split='val')

        nice_train_loader = DataLoader(he_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(he_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

    elif args.dataset == 'decathlon':
        nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)


    elif args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'LIDC':
        '''LIDC data'''
        # dataset = LIDC(data_path = args.data_path)
        dataset = MyLIDC(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'DDTI':
        '''DDTI data'''
        refuge_train_dataset = DDTI(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
        refuge_test_dataset = DDTI(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'Brat':
        '''Brat data'''
        dataset = Brat(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'STARE':
        '''STARE data'''
        # dataset = LIDC(data_path = args.data_path)
        dataset = STARE(args, data_path = args.data_path, transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'kits':
        '''kits data'''
        dataset = KITS(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'WBC':
        '''WBC data'''
        dataset = WBC(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'segrap':
        '''segrap data'''
        dataset = SegRap(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'toothfairy':
        '''toothfairy data'''
        dataset = ToothFairy(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'atlas':
        '''atlas data'''
        dataset = Atlas(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'pendal':
        '''pendal data'''
        dataset = Pendal(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    elif args.dataset == 'lnq':
        '''lnq data'''
        dataset = LNQ(args, data_path = args.data_path,transform = transform_train, transform_msk= transform_train_seg)

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.3 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=True)
        '''end'''

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader