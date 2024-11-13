import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import functional as FN

def spatial_argmax(logit):

  weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
  return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                      (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class PuckLocator(torch.nn.Module):
  def __init__(self, chans = [12,24,48,96]):
    super().__init__()

    arch = []

    self.chans = chans

    arch.append( torch.nn.BatchNorm2d(3) )
    arch.append( torch.nn.Conv2d(3, self.chans[0], 3, 2, 1) )
    arch.append( torch.nn.ReLU(True) )

    arch.append( torch.nn.BatchNorm2d(self.chans[0]) )
    arch.append( torch.nn.Conv2d(self.chans[0], self.chans[1], 3, 2, 1) )
    arch.append( torch.nn.ReLU(True) )

    arch.append( torch.nn.BatchNorm2d(self.chans[1]) )
    arch.append( torch.nn.Conv2d(self.chans[1], self.chans[2], 3, 2, 1) )
    arch.append( torch.nn.ReLU(True) )

    arch.append( torch.nn.BatchNorm2d(self.chans[2]) )
    arch.append( torch.nn.Conv2d(self.chans[2], self.chans[3], 3, 2, 1) )
    arch.append( torch.nn.ReLU(True) )

    self.cnet = torch.nn.Sequential(*arch)

    #arch.append( torch.nn.Conv2d(192, 1, 1) )

    self.puck_class = torch.nn.Linear(self.chans[3], 1)
    self.puck_aimpt = torch.nn.Conv2d(self.chans[3], 1, 1)

  def forward(self, img):
    output = self.cnet(img)

    return self.puck_class(output.mean(dim=[2,3])), spatial_argmax(self.puck_aimpt(output)[:,0])
    #return spatial_argmax(self.puck_aimpt(output)[:,0])

'''
def load_model():
    from torch import load
    from os import path
    r = PuckLocator()
    #r.load_state_dict(load(path.join('/content/pucklocator_SAS.th'), map_location='cpu'))
    r.load_state_dict(load(path.join('/content/pucklocator_SAS.th'), map_location='cpu'))
    return r
'''

def load_model():
    from torch import load
    from os import path
    r = PuckLocator()
    #r.load_state_dict(load(path.join('/content/pucklocator_SAS.th'), map_location='cpu'))
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'pucklocator_SAS.th'), map_location='cpu'))
    return r

def img_to_world_approx(img_pt, player_state, player):

    W2 = 0.3693891763687134 # height proxy for puck

    proj = np.array(player_state[player]['camera']['projection']).T
    view = np.array(player_state[player]['camera']['view']).T

    I0, I1 = img_pt[0], img_pt[1]

    Pr1, Pr2, Pr3, Pr4 = proj[0], proj[1], proj[2], proj[3]
    Vc1, Vc2, Vc3, Vc4 = view.T[0], view.T[1], view.T[2], view.T[3]

    A1 = np.dot(Pr4,Vc1)*I0 - np.dot(Pr1,Vc1)
    B1 = np.dot(Pr4,Vc2)*I0 - np.dot(Pr1,Vc2)
    C1 = np.dot(Pr4,Vc3)*I0 - np.dot(Pr1,Vc3)
    D1 = np.dot(Pr1,Vc4) - np.dot(Pr4,Vc4)*I0

    A2 = np.dot(Pr4,Vc1)*I1 + np.dot(Pr2,Vc1)
    B2 = np.dot(Pr4,Vc2)*I1 + np.dot(Pr2,Vc2)
    C2 = np.dot(Pr4,Vc3)*I1 + np.dot(Pr2,Vc3)
    D2 = -np.dot(Pr2,Vc4) - np.dot(Pr4,Vc4)*I1

    W3 = ( D2 - B2*W2 - (A2*D1) / A1 + (A2*B1*W2) / A1 ) / ( C2 - (A2*C1) / A1 )
    W1 = ( D1 - B1*W2 - C1*W3 ) / A1

    return [W1, W2, W3]

def steer_kart(kart_front, kart_center, puck):

    f_min_k = kart_front - kart_center
    p_min_f = puck - kart_front

    den = ( sum(p_min_f**2)**0.5 ) * ( sum(f_min_k**2)**0.5 )
    angle = np.arccos( np.dot(f_min_k, p_min_f) / den ) * (180/np.pi)

    # shift to [-1,1]

    new_angle = angle/90

    # left or right (cross product is position-agnostic):

    direction = np.cross(kart_front-kart_center, puck-kart_front)[1]

    new_angle = np.sign(direction)*new_angle

    #print('angle: ', new_angle)

    return new_angle


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        #from torch import load 
        from os import path 
        from torch import load

        self.team = None
        self.num_players = None

        # OLD:

        '''

        #self.ds_team20_image_agent = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'pucklocator_SAS.pt'))
        #self.ds_team20_image_agent = torch.load(path.join(path.dirname(path.abspath(__file__)), 'pucklocator_SAS.th'))
        self.ds_team20_image_agent = PuckLocator()
        #self.ds_team20_image_agent.load_state_dict(load("C:/Users/SSerrano/Documents/Deep Learning Course/PROJECT CMD/final/image_agent/pucklocator_SAS.th", map_location='cpu'))
        #What we were using: self.ds_team20_image_agent.load_state_dict(load("C:/Users/SSerrano/Documents/Deep Learning Course/PROJECT CMD/final/image_agent/pucklocator_SAS.th", map_location='cpu'))
        self.ds_team20_image_agent.load_state_dict(load("image_agent/pucklocator_SAS.th", map_location='cpu'))
        
        '''

        # OLD AGAIN:
        '''

        print('something')
        self.ds_team20_image_agent = PuckLocator()

        self.ds_team20_image_agent = torch.load(path.join(path.dirname(path.abspath(__file__)), 'pucklocator_SAS.th'), map_location='cpu')
        print('model loaded')
        print('model: ', self.ds_team20_image_agent)

        '''

        self.ds_team20_image_agent = load_model()
        #print('model loaded: ', self.ds_team20_image_agent)

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight
        #print('player_state: ', player_state)
        #print('player_image: ', player_image)

        ############
        ## Learnings:
        ## If we go too fast, we overshoot puck and don't turn around
        ## Make sure you go toward correct net! I think that matters (It's one you're facing at start of match)
        ## Have a rule that if near puck we don't go backward
        ############

        mod_inp1, mod_inp2 = FN.to_tensor(player_image[0])[None], FN.to_tensor(player_image[1])[None]
        comb_inp = torch.vstack( tuple( [mod_inp1, mod_inp2] ) )

        mod_out = self.ds_team20_image_agent(comb_inp)

        mod_out1, mod_out2, puck_in_front1, puck_in_front2 = mod_out[1][0], mod_out[1][1], mod_out[0][0].item(), mod_out[0][1].item()

        mod_out1, mod_out2 = img_to_world_approx(mod_out1.detach().numpy(), player_state, 0), img_to_world_approx(mod_out2.detach().numpy(), player_state, 1)

        front1, front2 = np.array(player_state[0]['kart']['front']), np.array(player_state[1]['kart']['front'])
        kart_loc1, kart_loc2 = np.array(player_state[0]['kart']['location']), np.array(player_state[1]['kart']['location'])
        mod_out1_arr, mod_out2_arr = np.array(mod_out1), np.array(mod_out2)

        dist1, dist2 = sum( (front1 - mod_out1_arr)**2.0 )**0.5, sum( (front2 - mod_out2_arr)**2.0 )**0.5

        # Compute steer angle:

        steer1 = 0.5
        steer2 = -0.8

        acceleration1 = 0
        acceleration2 = 0

        #acceleration1 = 0.2
        #acceleration2 = 0.2

        brake1 = True
        brake2 = True

        #brake1 = False
        #brake2 = False

        net_midpt = [-.025, 0.07, 64.5]

        # The following rule will also help us avoid going backward when we're near the puck
        # If we're at the puck, steer toward the net:
        
        if dist1 < 2.0:

          acceleration1 = 0.5
          brake1 = False

          steer1 = np.clip(3*steer_kart(front1, kart_loc1, net_midpt),-1,1)

        else:

          if puck_in_front1 > 0:

            steer1 = np.clip(5*steer_kart(front1, kart_loc1, mod_out1),-1,1)
            acceleration1 = 0.25
            brake1 = False

        if dist2 < 2.0:

          acceleration2 = 0.5
          brake2 = False

          steer2 = np.clip(3*steer_kart(front2, kart_loc2, net_midpt),-1,1)

        else:

          if puck_in_front2 > 0:

            steer2 = np.clip(5*steer_kart(front2, kart_loc2, mod_out2),-1,1)
            acceleration2 = 0.25
            brake2 = False

        # Should we have a max velocity? Clearly, accelerating too much has a large impact on ability to control kart's trajectory
        
        return [dict(acceleration=acceleration1, steer=steer1, brake=brake1), dict(acceleration=acceleration2, steer=steer2, brake=brake2)]
