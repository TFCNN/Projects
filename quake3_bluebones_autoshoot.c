/*
--------------------------------------------------
    James William Fletcher (james@voxdsp.com)
--------------------------------------------------

    This is a re-release with a crosshair and other
    minimal changes.
    
    This uses pre-computed weights designed to target
    only the aqua blue bones model.
    
    Original release:
    https://github.com/Quake3Aimbot/Perceptron-Autoshoot/blob/master/desperate.c
    
    Network described here:
    https://medium.com/swlh/training-a-neural-network-to-autoshoot-in-fps-games-e105f27ec1a0

    Tested using the ioQuake3 client.
    https://ioquake3.org/
    
    Compile:
    clang quake3_bluebones_autoshoot.c -Ofast -lX11 -lm -o aim

--------------------------------------------------
    Recommended Client Settings
--------------------------------------------------
    cg_oldRail "1"
    cg_noProjectileTrail "1"
    cg_forceModel "1"
    cg_railTrailTime "100"
    cg_crosshairSize "24"
    cg_draw2D "1"
    cg_gibs "0"
    cg_fov "160"
    cg_zoomfov "90"
    cg_brassTime "0"
    cg_drawCrosshair "7"
    cg_drawCrosshairNames "1"
    cg_marks "0"
    cg_crosshairPulse "1"
    xp_noParticles "1"
    xp_modelJump "0"
    xp_corpse "3"
    xp_crosshairColor "7"
    com_maxfps "1000"
    com_blood "0"
    cg_autoswitch "0"
    rate "25000"
    snaps "40"
    model "bones/default"
    headmodel "bones/default"
    team_model "bones/default"
    team_headmodel "bones/default"
    r_picmip "16"
    r_vertexLight "0"
    cg_shadows "0"
--------------------------------------------------
    ^ Thats how I like to configure my Quake 3 client.
--------------------------------------------------
*/

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

/***************************************************
   ~~ Pre-computed weights /
   These go in order from least trained weights to most trained weights.
*/
float pw[7][20][10] = {{{-0.000678,0.122750,0.877928},
{-0.014211,0.242387,0.769805},
{0.002608,0.360765,0.638191},
{-0.002566,0.221006,0.781539},
{-0.010689,0.275946,0.733333},
{0.003471,0.256959,0.739972},
{-0.011648,0.263964,0.740661},
{-0.005060,0.329051,0.675603},
{-0.000126,0.380449,0.619667},
{0.289186,0.062037,0.239718,0.207286},
{0.054666,0.330485,0.102977,0.304546},
{0.241590,0.246442,-0.162414,0.469380},
{0.211203,0.244579,0.036508,0.392616},
{0.498221,0.423368,0.166798},
{0.491952,0.382967,0.232922},
{0.434946,0.350813,0.258033},
{0.396584,0.359815,0.268922},
{0.474845,0.046089,0.503094},
{0.416468,0.368689,0.292179},
{-0.047015,-1.049416,0.317097,-0.296619,0.771759,0.097353,0.949131,1.860233,0.712723,0.454139}},

{{0.010719,0.543938,0.455386},
{-0.015895,0.655468,0.358169},
{0.005033,0.516414,0.483568},
{0.118469,0.564403,0.432609},
{0.003826,0.509000,0.490389},
{-0.001927,0.547794,0.452514},
{0.021021,0.582215,0.410340},
{0.009766,0.371914,0.626738},
{-0.036377,0.600968,0.431187},
{0.092139,0.421844,0.421788,0.283494},
{-0.002617,0.546351,0.121241,0.337279},
{0.182594,-0.000386,0.481288,0.441236},
{0.293086,0.115774,-0.006369,0.506366},
{0.336481,0.373318,0.317721},
{0.681950,0.013519,0.468137},
{0.834902,-0.050340,0.300285},
{0.433667,0.391089,0.439244},
{0.620147,0.207680,0.232108},
{0.485884,0.212128,0.354078},
{-0.856434,0.114034,-0.825134,0.117730,0.445520,0.670190,-0.199601,2.024588,0.272854,0.639285}},

{{0.000606,0.605079,0.394392},
{0.001021,0.643795,0.355214},
{0.001514,0.644063,0.354630},
{-0.067299,0.426359,0.583499},
{0.002358,0.525213,0.474127},
{0.001410,0.530144,0.469846},
{0.005478,0.624160,0.373757},
{0.018474,0.502543,0.491623},
{0.006355,0.547925,0.451326},
{0.274945,0.592837,0.240848,0.581794},
{0.439667,0.389202,0.395172,0.497968},
{0.570355,0.292174,0.692056,0.217283},
{0.539244,0.256297,0.393332,0.482979},
{0.681598,0.445250,0.610550},
{0.407798,0.330710,0.569185},
{0.593038,0.375053,0.433229},
{0.421818,0.129265,0.483957},
{0.535846,0.521908,0.155109},
{0.533411,0.521416,0.345681},
{-0.243367,-0.193479,0.263246,-0.091106,0.427943,0.162338,0.655157,0.049894,0.913666,0.235348}},

{{-0.249868,0.354957,0.605017},
{-0.248825,0.324373,0.574924},
{-0.306482,0.383830,0.513235},
{-0.387433,-0.042755,0.217665},
{-0.249585,0.249637,0.499999},
{-0.438810,0.123598,0.380656},
{-0.160687,0.439683,0.692259},
{-0.248685,0.248817,0.499914},
{-0.395345,0.095408,0.349948},
{0.239687,0.750243,0.390826,0.656487},
{0.012732,0.355883,0.495279,0.475360},
{0.338219,0.706628,0.726927,0.240757},
{0.203427,0.522732,0.319564,0.719768},
{0.788278,0.508291,0.765914},
{0.518362,0.056233,0.925399},
{0.740952,0.443612,0.498790},
{0.484976,0.580372,0.560987},
{0.858765,0.075200,0.741968},
{0.534240,0.307520,0.420081},
{0.071814,0.316577,0.295796,0.389007,0.653197,-0.551250,0.354053,0.134347,0.300965,0.625039}},

{{-0.501332,-0.119146,0.164647},
{-0.420861,-0.088981,0.093337},
{-0.488249,0.014249,0.063871},
{-0.294022,0.361086,0.611842},
{-0.345579,-0.064992,0.107459},
{-0.429214,-0.046138,0.046448},
{-0.234292,0.369881,0.615968},
{-0.435987,0.093112,0.082818},
{-0.293987,0.018881,-0.012355},
{0.125466,0.481611,0.847432,0.024421},
{-0.059559,0.629369,0.247038,0.523659},
{0.210344,0.494224,0.698030,0.708048},
{0.160262,0.300156,0.315606,0.590900},
{0.249587,0.373400,0.417000},
{0.850386,0.072037,0.641747},
{0.826178,0.407887,0.457404},
{0.068545,0.767936,0.739931},
{0.658893,0.007638,0.586034},
{0.301942,0.300436,0.571817},
{0.628862,-0.101672,-0.155959,-0.696697,-0.162159,0.141732,0.039324,1.024396,0.236846,0.503465}},

{{-0.078998,0.201827,0.820492},
{-0.355532,-0.083707,0.083707},
{0.001699,-0.064544,0.022921},
{-0.379071,-0.068936,0.068936},
{0.007576,-0.067579,0.025447},
{0.007328,-0.060154,-0.066911},
{0.003097,-0.046562,-0.062501},
{0.007253,-0.103037,-0.016908},
{0.001149,-0.139872,0.026408},
{0.492062,0.421604,0.479019,0.995845},
{0.632216,1.049548,0.637030,0.085617},
{0.650847,0.362372,0.399399,1.037698},
{0.739267,0.659977,-0.078243,0.173696},
{0.995847,0.569194,0.584373},
{1.043204,0.748504,0.298673},
{0.558819,0.552001,0.152801},
{0.995849,0.684686,0.437063},
{1.041229,0.574931,0.526867},
{0.878831,0.467534,0.110651},
{0.157062,0.213636,0.143775,-0.315940,0.283186,0.535890,0.039491,0.536632,0.338187,0.198256}},

{{-0.000024,0.000062,0.999940},
{-0,0.000001,0.999999},
{-0.000013,0.000079,0.999932},
{-0.000009,0.000031,0.999972},
{-0.000007,0.000027,0.999973},
{-0,0.000001,0.999999},
{-0,-0.013821,1.013821},
{-0.000007,0.000022,0.999978},
{-0.000008,0.000033,0.999967},
{0.448311,0.565085,0.185076,0.622112},
{0.389232,0.738935,0.592649,0.208486},
{0.294038,0.671509,0.666330,0.202099},
{0.240731,0.276413,0.267255,0.112908},
{0.788297,0.411209,0.813752},
{0.618703,0.483636,0.583136},
{0.732832,0.490073,0.569970},
{0.801913,0.293433,0.641857},
{0.591637,0.192624,0.715242},
{0.616393,-0.024514,0.574152},
{-0.983682,-0.305553,-1.308751,-0.929104,0.401342,2.611999,0.345443,1.314969,1.816733,1.092532}}};

/***************************************************
   ~~ Utils
*/
//https://www.cl.cam.ac.uk/~mgk25/ucs/keysymdef.h
//https://stackoverflow.com/questions/18281412/check-keypress-in-c-on-linux/52801588
int key_is_pressed(Display* dpy, KeySym ks)
{
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    int isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    return isPressed;
}

void speakS(const char* text)
{
    char s[256];
    sprintf(s, "/usr/bin/espeak \"%s\"", text);
    if(system(s) <= 0)
        sleep(1);
}

Window getWindow(Display* d, const int si) // gets child window mouse is over
{
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    XQueryPointer(d, RootWindow(d, si), &event.xbutton.root, &event.xbutton.window, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    event.xbutton.subwindow = event.xbutton.window;
    while(event.xbutton.subwindow)
    {
        event.xbutton.window = event.xbutton.subwindow;
        XQueryPointer(d, event.xbutton.window, &event.xbutton.root, &event.xbutton.subwindow, &event.xbutton.x_root, &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y, &event.xbutton.state);
    }
    const Window ret = event.xbutton.window;
    return ret;
}

/***************************************************
   ~~ Perceptron
*/
float _probability = 0.7; // minimum probability from neuron before attack

float doPerceptron(float* in, const uint32_t n, float* w)
{
    float ro = 0;
    for(size_t i = 0; i < n; i++)
        ro += in[i] * w[i];
    if(ro < 0){ro = 0;} // ReLU
    return ro;
}

float doDeepResult(float* in, const unsigned int k)
{
    //Output Array to Final Neuron
    #define outputs 10
    float h[outputs] = {0};

    //Quaterize the 3x3 into 2x2x4
    h[0] = doPerceptron((float[]){in[4], in[1], in[3], in[0]}, 4, pw[k][9]);
    h[1] = doPerceptron((float[]){in[4], in[1], in[2], in[5]}, 4, pw[k][10]);
    h[2] = doPerceptron((float[]){in[4], in[7], in[6], in[3]}, 4, pw[k][11]);
    h[3] = doPerceptron((float[]){in[4], in[7], in[8], in[5]}, 4, pw[k][12]);
    
    //3x3 to 1x3
    h[4] = doPerceptron((float[]){in[0], in[1], in[2]}, 3, pw[k][13]);
    h[5] = doPerceptron((float[]){in[3], in[4], in[5]}, 3, pw[k][14]);
    h[6] = doPerceptron((float[]){in[6], in[7], in[8]}, 3, pw[k][15]);

    //3x3 to 1x3
    h[7] = doPerceptron((float[]){in[0], in[3], in[6]}, 3, pw[k][16]);
    h[8] = doPerceptron((float[]){in[1], in[4], in[7]}, 3, pw[k][17]);
    h[9] = doPerceptron((float[]){in[2], in[5], in[8]}, 3, pw[k][18]);
    
    //Final neuron
    return doPerceptron(h, outputs, pw[k][19]);
}

/***************************************************
   ~~ Program Entry Point
*/
int main()
{
    printf("James William Fletcher (james@voxdsp.com)\n\n");
    printf("L-CTRL + L-ALT = Toggle BOT ON/OFF\n");
    printf("R-CTRL + R-ALT = Toggle HOTKEYS ON/OFF\n");
    printf("1/2 = How Desperate (1: Accurate / 2: Spray & Pray)\n");
    printf("P = Toggle crosshair\n\n");

    XColor c[9];
    Display *d;
    Window twin;
    GC gc = 0;
    int si;
    unsigned int x=0, y=0;
    XEvent event;
    memset(&event, 0x00, sizeof(event));
    unsigned int enable = 0;
    unsigned int offset = 3;
    unsigned int crosshair = 0;
    unsigned int hotkeys = 1;
    time_t ct = time(0);

    // open display 0
    d = XOpenDisplay(":0");
    if(d == NULL)
    {
        printf("Failed to open display\n");
        return 0;
    }

    // get default screen
    si = XDefaultScreen(d);

    // get graphics context
    gc = DefaultGC(d, si);
    
    while(1)
    {
        // loop every 1 ms (1,000 microsecond = 1 millisecond)
        usleep(1000);

        // inputs
        if(key_is_pressed(d, XK_Control_L) && key_is_pressed(d, XK_Alt_L))
        {
            if(enable == 0)
            {                
                enable = 1;
                usleep(300000);
                printf("BOT: ON\n");
                speakS("on");
            }
            else
            {
                enable = 0;
                usleep(300000);
                printf("BOT: OFF\n");
                speakS("off");
            }
        }
        
        // bot on/off
        if(enable == 1)
        {
            // get window
            twin = getWindow(d, si);

            // get center window point (x & y)
            XWindowAttributes attr;
            XGetWindowAttributes(d, twin, &attr);
            x = attr.width/2;
            y = attr.height/2;

            // set mouse event
            memset(&event, 0x00, sizeof(event));
            event.type = ButtonPress;
            event.xbutton.button = Button1;
            event.xbutton.same_screen = True;
            event.xbutton.subwindow = twin;
            event.xbutton.window = twin;


            // input toggle
            if(key_is_pressed(d, XK_Control_R) && key_is_pressed(d, XK_Alt_R))
            {
                if(hotkeys == 0)
                {
                    hotkeys = 1;
                    usleep(300000);
                    printf("HOTKEYS: ON [%ix%i]\n", x, y);
                    speakS("hk on");
                }
                else
                {
                    hotkeys = 0;
                    usleep(300000);
                    printf("HOTKEYS: OFF\n");
                    speakS("hk off");
                }
            }

            if(hotkeys == 1)
            {
                // crosshair toggle
                if(key_is_pressed(d, XK_P))
                {
                    if(crosshair == 0)
                    {
                        crosshair = 1;
                        usleep(300000);
                        printf("CROSSHAIR: ON\n");
                        speakS("cx on");
                    }
                    else
                    {
                        crosshair = 0;
                        usleep(300000);
                        printf("CROSSHAIR: OFF\n");
                        speakS("cx off");
                    }
                }

                if(key_is_pressed(d, XK_1))
                {
                    _probability = 0.7;
                    offset = 3;
                    printf("DESPERATION: OFF\n");
                    speakS("Confidence");
                }

                if(key_is_pressed(d, XK_2))
                {
                    _probability = 0;
                    offset = 0;
                    printf("DESPERATION: ON\n");
                    speakS("Desperation");
                }
            }

            //Get Image Block
            XImage *i = XGetImage(d, event.xbutton.window, x-1, y-1, 3, 3, AllPlanes, XYPixmap);
            if(i != NULL)
            {
                //Get Pixels
                c[0].pixel = XGetPixel(i, 0, 0);
                c[1].pixel = XGetPixel(i, 1, 0);
                c[2].pixel = XGetPixel(i, 2, 0);
                c[3].pixel = XGetPixel(i, 0, 1);
                c[4].pixel = XGetPixel(i, 1, 1);
                c[5].pixel = XGetPixel(i, 2, 1);
                c[6].pixel = XGetPixel(i, 0, 2);
                c[7].pixel = XGetPixel(i, 1, 2);
                c[8].pixel = XGetPixel(i, 2, 2);
                XFree(i);

                // colour map
                const Colormap map = XDefaultColormap(d, si);
                //https://thestarman.pcministry.com/asm/6to64bits.htm
                int ti = 0;
                for(int i = 0; i < 9; i++)
                    if(c[0].pixel >= 16777216)
                        ti = 1;
                if(ti == 1)
                {
                    XCloseDisplay(d);
                    continue;
                }

                //Test all "kernels"
                for(int k = offset; k < 7; ++k)
                {
                    //Compute Per Pixel Neuron outputs
                    float p[9]={0};
                    for(int i = 0; i < 9; i++)
                    {
                        XQueryColor(d, map, &c[i]);

                        const float r = c[i].red / 65535;
                        const float g = c[i].green / 65535;
                        const float b = c[i].blue / 65535;

                        p[i] = doPerceptron((float[]){r, g, b}, 3, pw[k][i]);
                    }

                    //Query Deep Result
                    const float deep_result = doDeepResult(p, k);

                    //If the neuron/perceptron says fire, fire !
                    if(deep_result > _probability)
                    {
                        // fire mouse down
                        event.type = ButtonPress;
                        event.xbutton.state = 0;
                        XSendEvent(d, PointerWindow, True, 0xfff, &event);
                        XFlush(d);
                        
                        // wait 100ms (or ban for suspected cheating)
                        usleep(100000);
                        
                        // release mouse down
                        event.type = ButtonRelease;
                        event.xbutton.state = 0x100;
                        XSendEvent(d, PointerWindow, True, 0xfff, &event);
                        XFlush(d);

                        // display ~1s recharge time
                        if(crosshair != 0)
                        {
                            crosshair = 2;
                            ct = time(0);
                        }

                        //Break the loop, we fired off a shot
                        break;
                    }
                }
            }

            if(crosshair == 1)
            {
                XSetForeground(d, gc, 65280);

                XDrawPoint(d, event.xbutton.window, gc, x-2, y-1);
                XDrawPoint(d, event.xbutton.window, gc, x-2, y);
                XDrawPoint(d, event.xbutton.window, gc, x-2, y+1);

                XDrawPoint(d, event.xbutton.window, gc, x+2, y-1);
                XDrawPoint(d, event.xbutton.window, gc, x+2, y);
                XDrawPoint(d, event.xbutton.window, gc, x+2, y+1);

                XFlush(d);
            }

            if(crosshair == 2)
            {
                if(time(0) > ct+1)
                    crosshair = 1;

                XSetForeground(d, gc, 16711680);

                XDrawPoint(d, event.xbutton.window, gc, x-2, y-1);
                XDrawPoint(d, event.xbutton.window, gc, x-2, y);
                XDrawPoint(d, event.xbutton.window, gc, x-2, y+1);

                XDrawPoint(d, event.xbutton.window, gc, x+2, y-1);
                XDrawPoint(d, event.xbutton.window, gc, x+2, y);
                XDrawPoint(d, event.xbutton.window, gc, x+2, y+1);

                XSetForeground(d, gc, 65280);

                XDrawPoint(d, event.xbutton.window, gc, x-3, y-1);
                XDrawPoint(d, event.xbutton.window, gc, x-3, y);
                XDrawPoint(d, event.xbutton.window, gc, x-3, y+1);

                XDrawPoint(d, event.xbutton.window, gc, x+3, y-1);
                XDrawPoint(d, event.xbutton.window, gc, x+3, y);
                XDrawPoint(d, event.xbutton.window, gc, x+3, y+1);

                XFlush(d);
            }
        }

        //
    }

    // done, never gets here in regular execution flow
    XCloseDisplay(d);
    return 0;
}

