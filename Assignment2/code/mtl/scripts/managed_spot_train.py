import os
import uuid
from datetime import datetime

import boto3

from mtl.scripts.train import main
from mtl.utils.config import command_line_parser


def get_newest_ckpt(s3_path):
    s3 = boto3.resource('s3')
    _, _, resume_bucket_name, resume_bucket_local_path = s3_path.split('/', 3)
    resume_bucket = s3.Bucket(resume_bucket_name)
    checkpoints = list(
        resume_bucket.objects.filter(Prefix=resume_bucket_local_path))
    checkpoints = [c for c in checkpoints if c.key.endswith(".ckpt")]
    if len(checkpoints) == 0:
        return None
    else:
        checkpoints = sorted(checkpoints, key=lambda x: x.last_modified, reverse=True)
        print(checkpoints)
        return f"s3://{checkpoints[0].bucket_name}/{checkpoints[0].key}"

if __name__ == '__main__':
    with open('aws_configs/default_s3_bucket.txt', 'r') as fh:
        S3_BUCKET_NAME = fh.read()
    with open('aws_configs/team_id.txt', 'r') as fh:
        TEAM_ID = int(fh.read())

    configs = [
        # You can specify the hyperparameters and the experiment name here.
        dict(
            log_dir='/home/ubuntu/results/',
            dataset_root='/home/ubuntu/miniscapes/',
            name='task4_branched_se_dec_berhu_testaug',
            optimizer='adam',
            optimizer_lr=0.0001,
            model_name='deeplab_branched_se',
            model_encoder_name='resnet50',
            resume='s3://dlad22-team19-ex2/T19_0510-0345_task4_branched_se_dec_berhu_9c208/last.ckpt'
            #batch_size=8,
            #num_epochs=32,
        )
        # If you want to run multiple experiments after each other, just add
        # further configs. Don't forget to check if the AWS timeout in
        # aws_start_instances.py is still sufficient.
        # dict(
        #     log_dir='/home/ubuntu/results/',
        #     dataset_root='/home/ubuntu/miniscapes/',
        #     name='Default',
        # )
    ]

    start_i = 0
    if os.path.isfile('instance_state.txt'):
        with open('instance_state.txt', 'r') as fh:
            states = fh.readlines()
        for s in reversed(states):
            last_i, last_resume = s.replace('\n','').split(',')
            last_i = int(last_i)
            if last_resume == 'finished':
                start_i = last_i + 1
                break
            else:
                start_i = last_i
                last_resume_ckpt = get_newest_ckpt(last_resume)
                if last_resume_ckpt is not None:
                    configs[last_i]['resume'] = last_resume_ckpt
                    print(f'Resume from {last_resume_ckpt}.')
                    break
                else:
                    print(f'{last_resume} does not exist.')

    for i in range(start_i, len(configs)):
        args = []
        for k, v in configs[i].items():
            args.extend([f'--{k}', str(v)])
        cfg = command_line_parser(args)
        print('Start config', i)
        print(cfg)
        timestamp = datetime.now().strftime('%m%d-%H%M')
        run_name = f'T{TEAM_ID:02d}_{timestamp}_{cfg.name}_{str(uuid.uuid4())[:5]}'
        s3_log_path = f"s3://{S3_BUCKET_NAME}/{run_name}/"
        with open('instance_state.txt', 'a') as fh:
            fh.write(f'{i},{s3_log_path}\n')
        main(cfg, run_name)
        with open('instance_state.txt', 'a') as fh:
            fh.write(f'{i},finished\n')
