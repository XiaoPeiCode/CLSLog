"""
LogHub dataset download and Drain-based log parsing.

Vendored from LogPAI Drain (MIT) via LogCAE preprocessing, adapted for standalone CLSLog.
"""

import os
import subprocess
import tarfile

from utils.drain import LogParser

LOGHUB_URLS = {
    'BGL': 'https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1',
    'Zookeeper': 'https://zenodo.org/record/3227177/files/Zookeeper.tar.gz?download=1',
}

PARSING_CONFIG = {
    'BGL': {
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [
            {'regex_pattern': r'core\.\d+', 'mask_with': 'CORE'},
            {'regex_pattern': r'blk_(|-)[0-9]+', 'mask_with': 'ID'},
            {'regex_pattern': r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', 'mask_with': 'IP'},
            {'regex_pattern': r'([0-9a-f]+[:][0-9a-f]+)', 'mask_with': 'Word'},
            {'regex_pattern': r'fpr[0-9]+[=]0x[0-9a-f]+ [0-9a-f]+ [0-9a-f]+ [0-9a-f]+', 'mask_with': 'FPR'},
            {'regex_pattern': r'r[0-9]+[=]0x[0-9a-f]+', 'mask_with': 'Word'},
            {'regex_pattern': r'[l|c|xe|ct]r=0x[0-9a-f]+', 'mask_with': 'Word'},
            {'regex_pattern': r'0x[0-9a-f]+', 'mask_with': 'Word'},
            {'regex_pattern': r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', 'mask_with': ' NUM'},
        ],
        'st': 0.5,
        'depth': 4,
    },
    'Zookeeper': {
        'log_format': r'<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [
            {'regex_pattern': r'(/|)(\d+\.){3}\d+(:\d+)?', 'mask_with': 'IP'},
        ],
        'st': 0.5,
        'depth': 4,
    },
}


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def download_dataset_archive(dataset_name, dataset_dir='dataset'):
    if dataset_name not in LOGHUB_URLS:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    root = _project_root()
    target_dir = os.path.join(root, dataset_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    archive_path = os.path.join(target_dir, f'{dataset_name}.tar.gz')
    log_path = os.path.join(target_dir, f'{dataset_name}.log')
    if os.path.exists(log_path):
        return target_dir

    if not os.path.exists(archive_path):
        url = LOGHUB_URLS[dataset_name]
        print(f'Downloading {dataset_name} from LogHub...')
        result = subprocess.run(
            ['curl', '-L', '--fail', '--progress-bar', '-o', archive_path, url],
            cwd=root,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f'Failed to download {dataset_name} dataset archive.')

    print(f'Extracting {archive_path}...')
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(target_dir)
    return target_dir


def parse_loghub_dataset(dataset_name, dataset_dir='dataset'):
    """Download (if needed) and parse a LogHub dataset with Drain."""
    if dataset_name not in PARSING_CONFIG:
        raise ValueError(f'Unsupported dataset for parsing: {dataset_name}')

    dataset_path = download_dataset_archive(dataset_name, dataset_dir)
    structured_path = os.path.join(dataset_path, f'{dataset_name}.log_structured.csv')
    if os.path.exists(structured_path):
        return structured_path

    cfg = PARSING_CONFIG[dataset_name]
    log_file = f'{dataset_name}.log'
    parser = LogParser(
        cfg['log_format'],
        indir=dataset_path,
        outdir=dataset_path,
        depth=cfg['depth'],
        st=cfg['st'],
        rex=cfg['regex'],
    )
    parser.parse(log_file)

    raw_log_path = os.path.join(dataset_path, log_file)
    if os.path.exists(raw_log_path):
        os.remove(raw_log_path)

    if not os.path.exists(structured_path):
        raise RuntimeError(f'Parsing finished but structured log not found: {structured_path}')
    return structured_path
