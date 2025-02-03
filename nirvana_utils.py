# Nirvana dependencies
try:
    import nirvana_dl
    from distutils.dir_util import copy_tree
except ImportError:
    nirvana_dl = None
import os
import transformers
import wandb

def copy_snapshot_to_out(out):
    """ The preempted run transfers its "state" to the restarted run through "snapshot path".
        "state" is a tar-archive that contains all files put into "snapshot path" by the preempted run.
        This function moves the files in the "state" archive to you local "out" dir.
    """
    if nirvana_dl and not int(os.environ.get("LOCAL_RANK") or 0):
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print("snapshot_path", snapshot_path)
        print("os.environ.get('SNAPSHOT_PATH')", os.environ.get("SNAPSHOT_PATH"))
        
        print(f"Copy the previous state from {snapshot_path} to {out}")

        print("\nsnapshot")
        os.system(f"ls {snapshot_path}")
        
        os.system(f"mv {snapshot_path}/auto_logs ./")
        if not os.path.isdir(out):
            os.system(f"mkdir -p {out}")
        os.system(f"cp -rf {snapshot_path}/* {out}")
        os.system(f"mv ./auto_logs {snapshot_path}")

        print("\nout")
        os.system(f"ls {out}")
    

def copy_out_to_snapshot(out, dump=True):
    """ This function copies all files in the local "out" directory to "snapshot path".
        dump: If True, put these files into tar-archive "state" and 
              send it to the Python DL output.  
    """
    if nirvana_dl and not int(os.environ.get("LOCAL_RANK") or 0):
        # snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        # print(f"Copy {out} to the snapshot path: {snapshot_path}")

        # # Delete previous state to avoid memory explosion
        # os.system(f"mv {snapshot_path}/auto_logs ./")
        # os.system(f"rm -rf {snapshot_path}/*")
        # os.system(f"mv ./auto_logs {snapshot_path}")
        # os.system(f"cp -rf {out}/* {snapshot_path}")

        # if dump:
        #     # Make it visible in the Python DL output
        #     nirvana_dl.snapshot.dump_snapshot(snapshot_path)
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy {out} to the snapshot path: {snapshot_path}")
        os.system(f"pip list --format=freeze > {out}/nirvana_requirements.txt")

        print("\nsnapshot before delete")
        os.system(f"ls {snapshot_path}")

        # Delete previous state to avoid memory explosion
        for filename in os.listdir(snapshot_path):
            print(filename, filename != "auto_logs")
            if filename != "auto_logs":
                os.system(f"rm -rf {snapshot_path}/{filename}")

        print("\nsnapshot before copy")
        os.system(f"ls {snapshot_path}")
        print("\nout before")
        os.system(f"ls {out}")

        os.system(f"cp -rf {out}/* {snapshot_path}")

        print("\nsnapshot after")
        os.system(f"ls {snapshot_path}")
        print("\nout after")
        os.system(f"ls {out}")

        if dump:
            # Make it visible in the Python DL output
            nirvana_dl.snapshot.dump_snapshot(snapshot_path)

        # for filename in os.listdir(snapshot_path):
        #     print(filename, filename != "auto_logs")
        #     if filename != "auto_logs":
        #         os.system(f"rm -rf {snapshot_path}/{filename}")
        
        # print("\nsnapshot after dump")
        # os.system(f"ls {snapshot_path}")


class TrainerNirvana(transformers.Trainer):
    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args, **kwargs)
        if self.is_local_process_zero() if self.args.save_on_each_node else self.is_world_process_zero():
            if 'wandb' in self.args.report_to:
                with open (f'{self.args.output_dir}/run_id.txt', 'w') as f:
                    f.write(f'{wandb.run.id}')
                os.system(f"ls {self.args.output_dir}")
                print('\n'*5)
                    
            copy_out_to_snapshot(self.args.output_dir)