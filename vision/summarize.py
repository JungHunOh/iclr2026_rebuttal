import os
import json
import glob

#for dataset in ['resisc45', 'cifar100', 'sun397', 'cars', 'dtd', 'cub200', 'food101']:
for model in ['vit-large', 'vit-base']:
    for dataset in ['cifar100', 'resisc45', 'food101', 'dtd', 'cub200', 'cars', 'sun397']:
        methods = ['base', 'pissa', 'dora', 'odlora', 'norslora']

        rs = [8,16,32]

        if dataset == 'cars' or dataset == 'sun397':
            lr = 5e-3
        else:
            lr = 2e-3
        
        if model == 'vit-large':
            lr *= 0.5

        dirs = []
        results = {}
        for method in methods:
            folders = glob.glob(f'./experiment/{dataset}/{model}*_{method}_*')
            for folder in folders:
                dirs.append(folder.split('/')[-1])
            
            results[method] = {}

        scales = []
        for method in methods:
            for r in rs:
                results[method]['r'+str(r)] = {}
                for j in range(4,5):
                    scale = j
                    results[method]['r'+str(r)][f'scale{scale}'] = []
                    scales.append(scale)

        for dir in sorted(dirs):
            try:
                split = dir.split('_')
                LR, scale, r, method = split[-4:-1]
                scale = scale.replace('alpha','scale')
                if float(scale[5:]) not in scales:
                    if method == 'norslora':
                        scale = 'scale'+str(int(scale[5:]) // int(r.replace('r','')) * 2)
                    else: continue
                if method not in methods: continue
                if int(r.replace('r','')) not in rs: continue
                #if 'bs256' not in dir: continue
                if abs(float(LR.replace('lr','')) - lr) != 0: continue
            except:
                continue
            
            for seed in range(1,4):
                json_file = os.path.join(f'./experiment/{dataset}',dir,f'final_eval_results_{seed}.json')
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    acc = float(data['eval_accuracy'])

                results[method][r][scale].append(acc)
                if len(results[method][r][scale]) == 3:
                    results[method][r][scale] = sum(results[method][r][scale]) / 3

        import matplotlib.pyplot as plt

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#e377c2','#7f7f7f','#bcbd22']
        #colors = {'base': colors[0], 'ours': colors[1], 'ourssvd1': colors[2], 'basesvd1': colors[3], 'oursdetachsvd1v3': colors[4], 'oursdetachv3': colors[5], 'ourssvdrv3': colors[6]}

        for rank in rs:
            for ii,method in enumerate(results.keys()):
                for r in results[method].keys():
                    for scale in results[method][r].keys():
                        if r != f'r{rank}': continue
                        acc = results[method][r][scale]

                        plt.scatter(float(scale.replace('scale','')),acc,color=colors[ii], marker='x', label=f'{method}_{r}')

            plt.grid(True, alpha=0.5)
            handles, labels = plt.gca().get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            plt.legend(unique.values(), unique.keys())
            plt.savefig(f'experiment/{dataset}_{model}_r{rank}.png')

            plt.cla()
            plt.clf()