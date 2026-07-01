import sys, os, datetime
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '.')

def ts(): return datetime.datetime.now().strftime('%H:%M:%S')

print('1 torch', ts()); sys.stdout.flush()
import torch
print('2 agent', ts()); sys.stdout.flush()
from agents.agent import Agent
print('3 social_graph', ts()); sys.stdout.flush()
from agents.social_graph import social_graph as build_social_graph, bsf as propagate_influence, signal_strenght as compute_signal_strength
print('4 scheduler', ts()); sys.stdout.flush()
from agents.scheduler import Scheduler
print('5 funnel', ts()); sys.stdout.flush()
from agents.funnel import Funnel_graph as FUNNEL_GRAPH, build as build_prerequisites, allowed as is_action_allowed
print('6 bc_trainer', ts()); sys.stdout.flush()
from agents.bc_trainer import build_vocab
print('7 rl_trainer', ts()); sys.stdout.flush()
from agents.rl_trainer import select_action_with_social, build_reverse_vocab
print('8 data', ts()); sys.stdout.flush()
from data.persona_cluster import assign_agent_personas
from data.loader import load_reviews, filter_, sort_split, build, dataset
print('ALL IMPORTS OK', ts()); sys.stdout.flush()
