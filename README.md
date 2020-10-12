# CarML-TCP

### Procedure to host code on GCP
1. Region: us-east4 (Northern Virginia) Zone: us-east4-c    

2. Machine Type: Compute-optimized: c2-standard-8 (8 vCPU, 32 GB memory)   

3. SSH into VM   
    
4. Install packages    
    `sudo apt-get -y install python3-pip`      
    `sudo apt install git-all`    
    `git clone https://github.com/YichengShen/CarML-CGC.git`   
    `cd CarML-CGC`   
    `pip3 install -r requirements.txt`   
    
5. Run
    `tmux`   
    `cd simulation_MXNET_final`   
    `nano config.yaml`  
    `python3 main.py --num-round 1`  
