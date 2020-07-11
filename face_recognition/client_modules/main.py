from client_service import ClientService
import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

def register(client_service):
    """
        Register new face to database
    """
    is_continue = True
    while is_continue:
        name = input("Name: ")
        client_service.run_register_new_face(name)
        if input("Do you want to add another FaceID? (y/n): ").lower() == 'n':
            is_continue = False
    client_service.dispose()

def demo(client_service):
    """
        Run demo Face Recognition
    """
    client_service.run_demo()

def main():
    print("1. Register")
    print("2. Demo")
    key = input("Your choice: ")
    client_service = ClientService()
    if key == '1':
        register(client_service)
    else:
        demo(client_service)
        

if __name__ == "__main__":
    main()