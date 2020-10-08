import cloud_server, rsu, vehicle


def main():
    model = [1, 2, 3, 4, 5]
    rsu_num = 5

    # Cloud server
    cloud = cloud_server.Cloud_Server(model)
    cloud.run_cloud_server()

    # RSU connects to cloud server as client
    rsu_list = []
    for rsu_id in range(rsu_num):
        new_rsu = rsu.RSU(rsu_id, cloud.ip, cloud.port)
        rsu_list.append(new_rsu)
    
    for each_rsu in rsu_list:
        each_rsu.run_rsu_as_client()
         



if __name__ == "__main__":
    main()