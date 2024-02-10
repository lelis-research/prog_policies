from __future__ import annotations
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import sys
from multiprocessing import Pool

sys.path.append('.')

import torch

from prog_policies.karel import KarelDSL
from prog_policies.karel_tasks import get_task_cls
from prog_policies.search_space import LatentSpace
from prog_policies.search_methods import HillClimbing, CEM, CEBS


if __name__ == '__main__':
    
    n_env = 16
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('task', help='Name of the task class')
    
    args = parser.parse_args()

    if args.task == "CleanHouse" or args.task == "StairClimberSparse" or args.task == "TopOff":
        sigma = 0.25
    elif args.task == "FourCorners" or args.task == "Harvester":
        sigma = 0.5
    elif args.task == "MazeSparse":
        sigma = 0.1
    else:
        sigma = 0.25
    
    dsl = KarelDSL()
    
    search_space = LatentSpace(dsl, sigma)
    
    env_args = {
        "env_height": 8,
        "env_width": 8,
        "crashable": False,
        "leaps_behaviour": True,
        "max_calls": 10000
    }
    
    search_method = HillClimbing(1000)
    
    if args.task == "CleanHouse":
        env_args["env_height"] = 14
        env_args["env_width"] = 22
    
    solutions = []
    if args.task == "DoorKey":
        solutions = [
            "DEF run m( turnLeft REPEAT R=2 r( turnLeft WHILE c( frontIsClear c) w( move w) r) turnLeft REPEAT R=3 r( REPEAT R=2 r( pickMarker move r) pickMarker turnLeft move pickMarker turnLeft REPEAT R=2 r( pickMarker move r) pickMarker turnRight move pickMarker turnRight r) WHILE c( not c( rightIsClear c) c) w( IF c( not c( frontIsClear c) c) i( turnLeft i) move w) turnRight move move turnRight WHILE c( noMarkersPresent c) w( move IF c( not c( frontIsClear c) c) i( turnLeft i) w) putMarker m)",
            "DEF run m( pickMarker move WHILE c( noMarkersPresent c) w( turnLeft move move w) REPEAT R=13 r( WHILE c( markersPresent c) w( turnRight pickMarker w) r) WHILE c( not c( markersPresent c) c) w( turnRight move w) WHILE c( markersPresent c) w( putMarker w) m)",
            "DEF run m( move REPEAT R=9 r( move IF c( not c( frontIsClear c) c) i( WHILE c( noMarkersPresent c) w( move turnLeft move w) pickMarker WHILE c( frontIsClear c) w( move w) WHILE c( noMarkersPresent c) w( move turnLeft w) putMarker i) r) m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( turnLeft pickMarker move move w) pickMarker move WHILE c( noMarkersPresent c) w( REPEAT R=17 r( move r) move turnLeft move w) putMarker m)",
            "DEF run m( turnRight WHILE c( noMarkersPresent c) w( turnRight move move w) pickMarker turnRight WHILE c( noMarkersPresent c) w( turnRight REPEAT R=12 r( move turnLeft move turnLeft r) move w) putMarker m)",
            "DEF run m( WHILE c( frontIsClear c) w( move w) WHILE c( noMarkersPresent c) w( move move turnLeft w) pickMarker REPEAT R=19 r( WHILE c( markersPresent c) w( putMarker w) move move turnLeft r) m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( turnLeft move move w) IF c( markersPresent c) i( pickMarker REPEAT R=18 r( turnLeft r) move i) WHILE c( noMarkersPresent c) w( move WHILE c( markersPresent c) w( turnLeft putMarker w) move turnRight w) m)",
            "DEF run m( REPEAT R=3 r( WHILE c( noMarkersPresent c) w( pickMarker REPEAT R=4 r( move r) move turnLeft w) r) pickMarker WHILE c( noMarkersPresent c) w( move turnLeft move w) putMarker m)"
            "DEF run m( WHILE c( noMarkersPresent c) w( WHILE c( noMarkersPresent c) w( turnLeft IF c( frontIsClear c) i( move turnRight move i) move w) pickMarker turnRight WHILE c( noMarkersPresent c) w( IF c( frontIsClear c) i( turnRight i) move w) putMarker w) m)",
            "DEF run m( REPEAT R=17 r( move IF c( rightIsClear c) i( pickMarker move i) pickMarker turnRight r) IFELSE c( noMarkersPresent c) i( WHILE c( noMarkersPresent c) w( turnRight move w) putMarker turnLeft turnLeft i) ELSE e( putMarker turnLeft move move e) m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( turnLeft move move w) pickMarker REPEAT R=15 r( move IF c( frontIsClear c) i( WHILE c( noMarkersPresent c) w( turnLeft move move w) putMarker i) turnRight r) m)",
            "DEF run m( turnRight IFELSE c( noMarkersPresent c) i( turnRight REPEAT R=2 r( pickMarker WHILE c( frontIsClear c) w( move w) WHILE c( noMarkersPresent c) w( move turnRight move w) turnRight r) putMarker i) ELSE e( move e) turnLeft m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( turnLeft move move w) REPEAT R=10 r( move pickMarker r) IF c( rightIsClear c) i( move WHILE c( noMarkersPresent c) w( turnLeft pickMarker move move w) putMarker WHILE c( markersPresent c) w( turnLeft w) putMarker i) m)",
            "DEF run m( REPEAT R=2 r( pickMarker WHILE c( noMarkersPresent c) w( REPEAT R=9 r( turnLeft move r) turnLeft turnLeft w) turnRight r) putMarker WHILE c( leftIsClear c) w( move putMarker move move w) WHILE c( rightIsClear c) w( move w) m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( move IF c( leftIsClear c) i( REPEAT R=3 r( move move move r) i) move turnLeft w) pickMarker turnLeft WHILE c( noMarkersPresent c) w( WHILE c( noMarkersPresent c) w( move turnRight move w) putMarker w) m)",
            "DEF run m( WHILE c( frontIsClear c) w( move w) turnLeft move WHILE c( noMarkersPresent c) w( turnRight move move w) IF c( leftIsClear c) i( pickMarker move move WHILE c( noMarkersPresent c) w( move turnRight move w) putMarker i) m)",
            "DEF run m( pickMarker WHILE c( noMarkersPresent c) w( turnLeft REPEAT R=14 r( move r) w) pickMarker WHILE c( noMarkersPresent c) w( move turnRight turnRight REPEAT R=11 r( move turnRight move r) w) putMarker WHILE c( frontIsClear c) w( pickMarker w) m)",
            "DEF run m( WHILE c( frontIsClear c) w( move move w) WHILE c( noMarkersPresent c) w( move move turnLeft w) turnRight IF c( markersPresent c) i( pickMarker WHILE c( noMarkersPresent c) w( move move turnLeft w) putMarker i) turnRight m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( turnLeft IF c( noMarkersPresent c) i( move i) move w) IF c( markersPresent c) i( pickMarker turnLeft move i) WHILE c( noMarkersPresent c) w( move turnRight move w) putMarker m)",
            "DEF run m( WHILE c( not c( markersPresent c) c) w( move move turnLeft w) WHILE c( frontIsClear c) w( pickMarker move w) pickMarker WHILE c( noMarkersPresent c) w( move turnLeft w) WHILE c( markersPresent c) w( putMarker turnLeft w) m)",
            "DEF run m( WHILE c( frontIsClear c) w( move w) turnRight WHILE c( noMarkersPresent c) w( turnLeft move w) pickMarker WHILE c( noMarkersPresent c) w( turnLeft REPEAT R=7 r( pickMarker r) IF c( noMarkersPresent c) i( move move i) w) putMarker m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( turnRight move IFELSE c( markersPresent c) i( pickMarker turnRight WHILE c( noMarkersPresent c) w( move move turnLeft w) putMarker i) ELSE e( pickMarker IF c( not c( frontIsClear c) c) i( move move i) e) w) m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( move IF c( frontIsClear c) i( WHILE c( noMarkersPresent c) w( turnRight move move w) pickMarker WHILE c( noMarkersPresent c) w( move turnRight move w) i) w) turnLeft turnLeft putMarker turnLeft m)",
            "DEF run m( move turnRight REPEAT R=18 r( pickMarker move IF c( rightIsClear c) i( turnRight i) WHILE c( rightIsClear c) w( move pickMarker turnLeft w) r) pickMarker WHILE c( noMarkersPresent c) w( pickMarker move turnRight w) putMarker m)",
            "DEF run m( REPEAT R=2 r( pickMarker move IF c( rightIsClear c) i( WHILE c( frontIsClear c) w( move move turnLeft turnRight w) i) IF c( frontIsClear c) i( move i) WHILE c( noMarkersPresent c) w( move turnRight move w) r) putMarker m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( move turnLeft move w) IF c( rightIsClear c) i( pickMarker turnLeft i) IF c( noMarkersPresent c) i( move WHILE c( noMarkersPresent c) w( IF c( frontIsClear c) i( turnRight i) move w) putMarker move move i) m)",
            "DEF run m( move WHILE c( noMarkersPresent c) w( turnLeft move move w) pickMarker move IF c( noMarkersPresent c) i( move turnLeft WHILE c( frontIsClear c) w( move w) turnLeft WHILE c( noMarkersPresent c) w( turnRight move w) putMarker i) m)",
            "DEF run m( WHILE c( noMarkersPresent c) w( pickMarker turnLeft move move w) turnRight IFELSE c( markersPresent c) i( pickMarker i) ELSE e( turnLeft turnLeft e) WHILE c( noMarkersPresent c) w( move turnRight move move move move w) REPEAT R=3 r( putMarker r) m)"
        ]
    elif args.task == "OneStroke":
        solutions = []
    elif args.task == "Seeder":
        solutions = []
    elif args.task == "Snake":
        solutions = []

    for init_prog_str in solutions:
        
        init_prog = dsl.parse_str_to_node(init_prog_str)
        init_prog_intseq = search_space.leaps_dsl.str2intseq(init_prog_str)
        init_prog_len = torch.tensor([len(init_prog_intseq)])
        init_prog_intseq += [dsl.t2i['<pad>']] * (45 - init_prog_len)
        init_prog_torch = torch.tensor(init_prog_intseq, dtype=torch.long, device=search_space.torch_device).unsqueeze(0)
        _, init_enc_hxs = search_space.latent_model.vae.encoder(init_prog_torch, init_prog_len)
        init_latent = search_space.latent_model.vae._sample_latent(init_enc_hxs.squeeze(0))
        init_latent = init_latent.squeeze(0)
        
        init_decoded = search_space._decode(init_latent)
        
        init = init_latent, init_decoded
        
        print('Input program:', init_prog_str)
        print('Decoded program:', dsl.parse_node_to_str(init_decoded))
        
        task_cls = get_task_cls(args.task)
        task_envs = [task_cls(env_args, i) for i in range(n_env)]
        
        progs, rewards = search_method.search(search_space, task_envs, 0, 1000, init)
        print(rewards)
        print([dsl.parse_node_to_str(p) for p in progs])
        print()
