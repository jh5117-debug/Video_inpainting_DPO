# VideoPainter DAVIS Eval Set Audit

- davis_root: `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`
- image_dir: `/mnt/workspace/hj/nas_hj/data/external/davis_432_240/JPEGImages_432_240`
- mask_dir: `/mnt/workspace/hj/nas_hj/data/external/davis_432_240/test_masks`
- available paired videos: 50
- selected videos: 50
- selected names: bear, blackswan, bmx-bumps, bmx-trees, boat, breakdance, breakdance-flare, bus, camel, car-roundabout, car-shadow, car-turn, cows, dance-jump, dance-twirl, dog, dog-agility, drift-chicane, drift-straight, drift-turn, elephant, flamingo, goat, hike, hockey, horsejump-high, horsejump-low, kite-surf, kite-walk, libby, lucia, mallard-fly, mallard-water, motocross-bumps, motocross-jump, motorbike, paragliding, paragliding-launch, parkour, rhino, rollerblade, scooter-black, scooter-gray, soapbox, soccerball, stroller, surf, swing, tennis, train
- eval is full DAVIS50: True
- frame convention: each clip trimmed to 4k+1 frames, capped at 49
- resize for VideoPainter inference: 720x480
