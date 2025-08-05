import pandas as pd

# Data to be added/updated
# Keywords are stored as a comma-separated string
data_to_add = [
    {
        "label": "Agricultural Equipment Services",
        "definition": "This refers to businesses that provide maintenance, repair, sales, or rental services for machinery and tools used in farming and agricultural operations. This can range from tractors and harvesters to irrigation systems and specialized attachments.",
        "keywords": "farm machinery, tractor repair, harvester maintenance, agricultural implement, equipment sales, equipment rental, precision agricultural technology servicing"
    },
    {
        "label": "Soil Nutrient Application Services",
        "definition": "This describes services that specialize in the assessment and application of fertilizers, organic matter, or other amendments to improve soil fertility and optimize crop yield. This includes soil testing and tailored nutrient spreading.",
        "keywords": "fertilizer application, soil enrichment, crop nutrition, custom spreading, soil testing, nutrient management plan, organic soil inputs"
    },
    {
        "label": "Pesticide Application Services",
        "definition": "This pertains to services offering the controlled and targeted application of chemical or biological substances to protect crops and agricultural land from pests, weeds, and diseases. This often requires specialized knowledge and equipment.",
        "keywords": "crop spraying, pest management, herbicide application, insecticide services, fungicide treatment, agricultural pest control, applicator license"
    },
    {
        "label": "Ornamental Plant Nurseries",
        "definition": "Businesses that cultivate, grow, and sell plants primarily for decorative or aesthetic purposes, including flowers, shrubs, trees, and houseplants. They may also offer related gardening supplies and advice.",
        "keywords": "plant nursery, garden center, decorative plants, horticulture, landscaping plants, shrubs for sale, tree saplings"
    },
    {
        "label": "Landscaping Services",
        "definition": "Companies providing comprehensive design, installation, and maintenance services for outdoor spaces such as lawns, gardens, and grounds for residential, commercial, or public properties. This often includes both softscaping (plants) and hardscaping (patios, walkways).",
        "keywords": "garden design, lawn care, landscape installation, grounds maintenance, hardscaping services, softscaping, outdoor improvement"
    },
    {
        "label": "Gardening Services",
        "definition": "Services focused on the routine care and cultivation of gardens, typically involving planting, weeding, pruning, mulching, and general garden upkeep for homeowners or smaller properties.",
        "keywords": "garden maintenance, planting services, weeding services, residential gardening, flower bed care, vegetable garden upkeep, soil management"
    },
    {
        "label": "Tree Services - Pruning / Removal",
        "definition": "Specialized arboricultural services offering the professional pruning of trees for health, safety, and aesthetics, as well as the safe and efficient removal of dead, diseased, hazardous, or unwanted trees.",
        "keywords": "tree trimming, tree cutting, arborist services, stump grinding, emergency tree removal, tree health care, branch pruning"
    },
    {
        "label": "Veterinary Services",
        "definition": "Professional medical services provided by licensed veterinarians for the health and well-being of animals. This includes diagnosis, treatment of illnesses and injuries, surgery, and preventative care for various animal species.",
        "keywords": "animal healthcare, veterinarian practice, animal medical treatment, pet wellness, livestock health services, surgical procedures for animals, animal diagnostics"
    },
    {
        "label": "Veterinary Clinics",
        "definition": "Physical facilities or establishments where veterinarians and their staff provide a range of medical and surgical treatments for animals. These clinics are equipped for examinations, diagnostics, and procedures.",
        "keywords": "animal hospital, pet clinic, veterinary practice facility, animal medical center, small animal care clinic, large animal care facility, vet office"
    },
    {
        "label": "Pet Boarding Services",
        "definition": "Businesses offering temporary lodging, care, and supervision for domestic pets when their owners are away. Services typically include feeding, exercise, safe enclosure, and sometimes additional amenities like grooming or playtime.",
        "keywords": "kennel services, pet hotel, dog boarding, cat boarding, pet sitting facility, animal lodging, overnight pet care"
    },
    {
        "label": "Animal Day Care Services",
        "definition": "Businesses offering supervised care for pets, typically dogs, during daytime hours while owners are at work or otherwise occupied. Services often include socialization, exercise, and basic monitoring.",
        "keywords": "doggy day care, pet sitting, canine socialisation, supervised pet play, daily pet care, animal supervision, dog walking services"
    },
    {
        "label": "Pet Grooming Services",
        "definition": "Professional services for cleaning and maintaining the appearance and hygiene of domestic pets. This typically includes bathing, brushing, hair trimming, nail clipping, and ear cleaning.",
        "keywords": "dog grooming, cat grooming, pet stylists, animal bathing, pet hygiene, nail trimming for pets, pet beautification"
    },
    {
        "label": "Animal Training Services",
        "definition": "Services providing instruction and behavior modification for animals, most commonly dogs, to teach obedience, specific skills, or address behavioral issues. This can be done through group classes or individual sessions.",
        "keywords": "dog training, pet obedience, animal behaviorist, puppy classes, canine skills training, trick training, problem behavior correction"
    },
    {
        "label": "Veterinary Health Centers",
        "definition": "Comprehensive facilities offering a wide range of veterinary medical services, potentially including advanced diagnostics, specialized treatments, emergency care, and wellness programs for various animal species. These are often larger than standard clinics.",
        "keywords": "animal medical center, veterinary hospital, advanced pet care, animal surgery center, pet diagnostics, emergency vet, multi-vet practice"
    },
    {
        "label": "Animal Trainers",
        "definition": "Individuals or businesses specializing in the education and behavior modification of animals. This can encompass a wide range of animals and training disciplines, from domestic pets to working animals or performance animals.",
        "keywords": "professional animal trainer, dog obedience instructor, horse trainer, exotic animal trainer, animal behavior specialist, clicker training, positive reinforcement training"
    },
    {
        "label": "Livestock Dealer Services",
        "definition": "Businesses or individuals engaged in the buying, selling, and trading of farm animals such as cattle, sheep, pigs, and poultry. They act as intermediaries between producers and buyers or other dealers.",
        "keywords": "cattle trading, livestock brokerage, farm animal sales, livestock auction services, animal procurement, livestock market, poultry dealing"
    },
    {
        "label": "Timber Harvesting Operations",
        "definition": "Activities involving the felling of trees, processing them at the logging site (for example, delimbing, bucking), and transporting logs or timber to mills or other destinations. This requires specialized equipment and adherence to forestry regulations.",
        "keywords": "logging operations, tree felling, forest harvesting, timber extraction, sustainable forestry, log transport, wood procurement"
    },
    {
        "label": "Fishing and Hunting Services",
        "definition": "Commercial services that guide or outfit individuals or groups for recreational fishing or hunting expeditions. This can include providing equipment, access to private lands or waters, and expertise in locating game or fish.",
        "keywords": "guided fishing trips, hunting outfitters, charter fishing, game hunting guides, angling tours, sport fishing, wildlife excursions"
    },
    {
        "label": "Well Maintenance Services",
        "definition": "Services focused on the upkeep, repair, and cleaning of water wells, oil wells, or geothermal wells to ensure their proper functioning, efficiency, and longevity. This can include pump service, well cleaning, and system inspections.",
        "keywords": "water well servicing, well pump repair, well cleaning, borehole maintenance, well inspection, oil well workover, geothermal well care"
    },
    {
        "label": "Field Welding Services",
        "definition": "Mobile welding services performed on-site at a client\'s location, rather than in a workshop. This is often required for large structures, pipelines, heavy equipment repair, or construction projects where materials cannot be easily transported.",
        "keywords": "mobile welding, on-site fabrication, pipeline welding, structural steel welding, equipment repair welding, portable welding, emergency welding"
    },
    {
        "label": "Sand and Gravel Mining",
        "definition": "The extraction of sand, gravel, and crushed stone from open pits or quarries. These materials are primarily used in construction for making concrete, asphalt, and road base.",
        "keywords": "aggregate extraction, quarry operations, crushed stone production, construction aggregates, sand pit, gravel excavation, mining raw materials"
    },
    {
        "label": "Residential Driveway Construction",
        "definition": "Services specializing in the design, installation, and repair of driveways for private homes. Materials commonly used include asphalt, concrete, pavers, or gravel.",
        "keywords": "home driveway paving, concrete driveway installation, asphalt paving residential, driveway repair, paver driveways, gravel driveway services, new driveway construction"
    },
    {
        "label": "Commercial Driveway Construction",
        "definition": "Services focused on the construction and maintenance of driveways, parking lots, and access roads for commercial properties, industrial sites, and public facilities. These often involve larger scale projects and heavier-duty materials.",
        "keywords": "commercial paving, parking lot construction, asphalt services commercial, concrete paving business, industrial driveways, access road building, heavy-duty surfacing"
    },
    {
        "label": "Fencing Construction Services",
        "definition": "Businesses that design, supply, and install various types of fences for residential, commercial, agricultural, or industrial properties. Materials can include wood, vinyl, metal, chain-link, or composite.",
        "keywords": "fence installation, fence building, residential fencing, commercial fencing contractors, security fencing, agricultural fencing, custom fence design"
    },
    {
        "label": "Sidewalk Construction Services",
        "definition": "Services involved in the installation, repair, and replacement of public or private sidewalks and walkways. This typically involves concrete work and ensuring compliance with accessibility standards.",
        "keywords": "concrete sidewalk paving, walkway installation, footpath construction, sidewalk repair services, ADA compliant sidewalks, pedestrian pathway building, municipal sidewalk projects"
    },
    {
        "label": "Commercial Irrigation Systems",
        "definition": "Design, installation, and maintenance of large-scale watering systems for commercial properties such as business parks, sports fields, golf courses, and agricultural operations. These systems aim for efficient water distribution.",
        "keywords": "commercial sprinklers, large-scale irrigation, agricultural irrigation, golf course watering systems, smart irrigation commercial, drip irrigation systems, water management solutions"
    },
    {
        "label": "Residential Drainage Systems",
        "definition": "Installation and maintenance of systems designed to manage excess surface water and groundwater around homes and residential properties. This includes French drains, sump pumps, gutter downspout connections, and yard grading.",
        "keywords": "yard drainage solutions, French drain installation, sump pump services, gutter runoff management, landscape drainage, water damage prevention home, surface water control"
    },
    {
        "label": "Residential Snow Removal",
        "definition": "Services providing snow clearing for private homes, typically including driveways, walkways, and steps. This can be offered on a per-visit or seasonal contract basis.",
        "keywords": "home snow plowing, driveway snow clearing, sidewalk shoveling residential, snow blowing services, residential snow management, ice removal home"
    },
    {
        "label": "Commercial Snow Removal",
        "definition": "Snow and ice management services for commercial properties, including parking lots, access roads, sidewalks, and entryways to ensure safety and accessibility for employees and customers. Often involves larger equipment and 24/7 availability.",
        "keywords": "commercial snow plowing, parking lot snow clearing, de-icing services, snow hauling commercial, business snow removal, ice management services, winter property maintenance"
    },
    {
        "label": "General Snow Removal Services",
        "definition": "Encompasses a broad range of snow clearing and ice management activities for various types of properties, including residential, commercial, and potentially municipal areas. This label can cover both plowing, blowing, shoveling, and de-icing.",
        "keywords": "snow plowing services, snow clearing, ice control, snow management, winter services, snow shoveling, de-icing treatments"
    },
    {
        "label": "Land Leveling Services",
        "definition": "Services involving the reshaping of land surfaces to a desired grade or slope, typically for agricultural purposes, construction site preparation, or improved drainage. This often uses heavy machinery like bulldozers and scrapers.",
        "keywords": "earthmoving, grading services, site preparation, agricultural land shaping, construction leveling, terrain smoothing, precision land forming"
    },
    {
        "label": "Residential Drain Cleaning",
        "definition": "Services focused on clearing blockages and maintaining the proper function of drains and sewer lines within private homes. This includes kitchen sinks, bathroom drains, and main sewer line cleaning.",
        "keywords": "home drain unblocking, sewer cleaning residential, clogged pipe service, plumbing drain service, rooter service, hydro jetting residential, sink drain clearing"
    },
    {
        "label": "Commercial Drain Cleaning",
        "definition": "Specialized drain and sewer line cleaning services for businesses, industrial facilities, and multi-unit residential buildings. This often involves handling larger pipes, more complex systems, and grease traps.",
        "keywords": "business drain services, industrial sewer cleaning, grease trap maintenance, commercial plumbing drain, hydro jetting commercial, storm drain cleaning, preventative drain maintenance"
    },
    {
        "label": "Street Cleaning Operations",
        "definition": "Municipal or private services responsible for the regular cleaning of public streets, roads, and sometimes parking lots. This typically involves mechanical sweepers to remove debris, dirt, and litter.",
        "keywords": "road sweeping, municipal cleaning services, power sweeping, street sweeper services, debris removal streets, parking lot sweeping, urban sanitation"
    },
    {
        "label": "Conveyor System Installation",
        "definition": "Services specializing in the design, assembly, and implementation of conveyor systems used for transporting materials or goods within industrial, manufacturing, commercial, or logistical facilities.",
        "keywords": "material handling systems, automated conveyors, production line setup, warehouse automation, baggage handling systems, assembly line installation, belt conveyor systems"
    },
    {
        "label": "Low-Rise Signage Installation",
        "definition": "Installation of signs on buildings or structures that are generally a few stories high (for example, up to 3-4 stories). This can include storefront signs, monument signs, and directional signage, typically not requiring specialized high-access equipment.",
        "keywords": "storefront sign fitting, business signage installation, ground-level signs, retail sign mounting, building facade signs, small building sign installation, non-illuminated sign installation"
    },
    {
        "label": "High-Rise Signage Installation",
        "definition": "Specialized installation of large-format signs on tall buildings and skyscrapers, requiring specialized access equipment like cranes, swing stages, or rappelling gear, along with stringent safety protocols.",
        "keywords": "skyscraper signs, tall building signage, crane lift sign installation, rappelling sign installation, large format banner installation, building wrap signs, high-access sign fitting"
    },
    {
        "label": "Tank Installation Services",
        "definition": "Services involved in the placement, setup, and connection of various types of storage tanks, such as water tanks, fuel tanks, chemical tanks, or septic tanks, for residential, commercial, or industrial use.",
        "keywords": "storage tank setup, fuel tank installation, water storage solutions, septic system installation, industrial tank fitting, chemical storage tanks, underground tank installation"
    },
    {
        "label": "Residential Communication Equipment Installation",
        "definition": "Installation of telecommunications and data equipment in private homes, such as internet modems, routers, satellite dishes, TV antennas, and home networking cabling.",
        "keywords": "home internet setup, satellite TV installation, home networking, residential telecom services, Wi-Fi installation, cable TV fitting, antenna installation"
    },
    {
        "label": "Commercial Communication Equipment Installation",
        "definition": "Installation and configuration of communication systems for businesses, including structured cabling, business phone systems (VoIP), data networks, server racks, and audiovisual conferencing equipment.",
        "keywords": "business phone systems, structured cabling installation, data center setup, office network installation, Voice over Internet Protocol services, commercial AV systems, server rack installation"
    },
    {
        "label": "Low-Rise Glass Installation",
        "definition": "Installation and replacement of glass in windows, doors, and storefronts for buildings typically up to a few stories high. This includes single-pane, double-pane, and safety glass for residential and small commercial properties.",
        "keywords": "residential window glass, storefront glass replacement, commercial glazing low-rise, door glass installation, window pane repair, small building glass fitting, double glazing installation"
    },
    {
        "label": "High-Rise Glass Installation",
        "definition": "Specialized glazing services for tall buildings and skyscrapers, involving the installation and replacement of curtain walls, large window panes, and architectural glass, requiring cranes, swing stages, or other high-access methods.",
        "keywords": "skyscraper glazing, curtain wall installation, commercial high-rise windows, architectural glass fitting, structural glazing, high-access glass replacement, building facade glazing"
    },
    {
        "label": "Industrial Machinery Installation",
        "definition": "Services focused on the setup, assembly, alignment, and commissioning of heavy machinery and equipment used in manufacturing plants, production facilities, and other industrial settings.",
        "keywords": "factory equipment setup, heavy machine rigging, manufacturing line installation, plant machinery fitting, equipment commissioning, production machinery installation, millwright services"
    },
    {
        "label": "Agricultural Machinery Installation",
        "definition": "Setting up and commissioning specialized farming equipment on-site, such as irrigation systems, milking parlors, automated feeding systems, and large crop processing machinery.",
        "keywords": "farm equipment setup, irrigation system installation, dairy equipment fitting, automated farm systems, crop processing machinery, agricultural robotics installation, precision agricultural equipment setup"
    },
    {
        "label": "Grain Handling Machinery Installation",
        "definition": "Installation of equipment used for moving, storing, and processing grains, such as augers, conveyors, grain elevators, silos, and grain dryers in agricultural or storage facilities.",
        "keywords": "grain elevator construction, silo installation, grain auger setup, conveyor systems for grain, grain drying equipment, bulk material handling farm, feed mill machinery"
    },
    {
        "label": "Dock and Pier Construction",
        "definition": "Design and construction of structures extending from land out over water, such as boat docks, fishing piers, loading docks for marine vessels, and boardwalks. This involves working with marine environments and materials resistant to water.",
        "keywords": "marine construction, boat dock building, pier installation, waterfront structures, piling installation, boardwalk construction, floating dock systems"
    },
    {
        "label": "Road and Highway Construction",
        "definition": "Large-scale civil engineering projects involving the building of new roads, highways, expressways, and interchanges. This includes earthwork, paving, drainage, and installation of related infrastructure like bridges and traffic signals.",
        "keywords": "highway building, new road development, civil engineering construction, asphalt paving infrastructure, concrete road construction, infrastructure projects, freeway construction"
    },
    {
        "label": "Road Maintenance Services",
        "definition": "Ongoing upkeep and repair of existing roads and highways, including pothole patching, resurfacing, crack sealing, line striping, and guardrail repair to ensure safety and extend pavement life.",
        "keywords": "highway upkeep, pothole repair, asphalt resurfacing, road patching, pavement preservation, line marking services, guardrail maintenance"
    },
    {
        "label": "Pipeline Construction Services",
        "definition": "Services related to the planning, trenching, laying, welding, and testing of pipelines used for transporting oil, gas, water, or other fluids over various distances and terrains.",
        "keywords": "oil pipeline installation, gas pipeline laying, water main construction, utility pipeline services, trenching for pipelines, pipeline welding, cross-country pipelines"
    },
    {
        "label": "New Ground Pipeline Installation",
        "definition": "Specifically refers to the construction of entirely new pipeline systems, often in previously undeveloped areas or along new routes, as opposed to replacing or repairing existing lines.",
        "keywords": "new pipeline projects, greenfield pipeline construction, pipeline routing, right-of-way clearing pipeline, new oil and gas infrastructure, water transmission lines, slurry pipeline installation"
    },
    {
        "label": "Excavation Services",
        "definition": "Services involving digging, earthmoving, and trenching for various purposes such as foundation preparation, utility line installation, landscaping, and site development. This uses heavy equipment like excavators and backhoes.",
        "keywords": "digging services, earthmoving contractors, site excavation, trenching services, foundation digging, land clearing, backhoe services"
    },
    {
        "label": "Residential Plumbing Services",
        "definition": "Plumbing services for private homes, including installation, repair, and maintenance of water supply lines, drainage systems, fixtures (sinks, toilets, showers), and water heaters.",
        "keywords": "home plumbing repair, fixture installation, leaky pipe repair, water heater service residential, toilet repair, sink installation, emergency plumber home"
    },
    {
        "label": "Commercial Plumbing Services",
        "definition": "Plumbing services tailored for businesses, office buildings, retail spaces, and other commercial properties. This includes installation and maintenance of larger, more complex plumbing systems, restrooms, and specialized fixtures.",
        "keywords": "business plumbing solutions, office building plumbing, retail plumbing services, commercial water heaters, public restroom plumbing, grease trap plumbing, backflow prevention commercial"
    },
    {
        "label": "Industrial Plumbing Services",
        "definition": "Specialized plumbing services for manufacturing plants, factories, and other industrial facilities, dealing with process piping, high-pressure systems, industrial waste drainage, and specialized fluid handling systems.",
        "keywords": "factory plumbing, process piping installation, industrial pipefitting, plant maintenance plumbing, high-pressure water systems, wastewater systems industrial, specialized fluid handling"
    },
    {
        "label": "Boiler Installation Services",
        "definition": "Services for the setup and commissioning of new boiler systems used for heating or industrial processes in residential, commercial, or industrial settings. This includes connecting to fuel, water, and exhaust systems.",
        "keywords": "new boiler setup, heating system installation, steam boiler fitting, hot water boiler installation, commercial boiler replacement, industrial boiler commissioning, boiler system integration"
    },
    {
        "label": "Boiler Repair Services",
        "definition": "Maintenance and repair services for existing boiler systems, addressing issues like leaks, no heat, pressure problems, or component failure to restore functionality and efficiency.",
        "keywords": "boiler maintenance, heating system repair, emergency boiler service, steam boiler troubleshooting, hot water boiler repair, commercial boiler upkeep, industrial boiler servicing"
    },
    {
        "label": "Steam Services",
        "definition": "Services related to the installation, maintenance, and repair of steam generation and distribution systems, including steam boilers, steam pipes, traps, and related components used for heating or industrial processes.",
        "keywords": "steam system installation, steam pipe fitting, boiler steam repair, industrial steam solutions, steam trap maintenance, steam heating services, process steam systems"
    },
    {
        "label": "Gas Installation Services",
        "definition": "Services for installing and connecting natural gas or propane lines and appliances in residential, commercial, or industrial properties. This includes leak testing and ensuring compliance with safety codes.",
        "keywords": "natural gas line fitting, propane tank installation, gas appliance hookup, gas pipe installation, commercial gas services, residential gas fitting, gas leak detection"
    },
    {
        "label": "Medical Gas Installation Services",
        "definition": "Specialized installation of piped gas systems in hospitals, clinics, and other healthcare facilities for delivering medical gases like oxygen, nitrous oxide, medical air, and vacuum to patient care areas and operating rooms.",
        "keywords": "hospital gas pipeline, oxygen system installation healthcare, medical air systems, nitrous oxide piping, vacuum system medical, healthcare facility gas, certified medical gas installer"
    },
    {
        "label": "Fire Protection System Services",
        "definition": "Design, installation, maintenance, and inspection of fire suppression and alarm systems, including sprinklers, fire alarms, standpipes, and fire extinguishers, for residential, commercial, and industrial buildings.",
        "keywords": "fire sprinkler installation, fire alarm systems, fire suppression maintenance, life safety systems, fire extinguisher services, commercial fire protection, industrial fire safety"
    },
    {
        "label": "HVAC Installation and Service",
        "definition": "Services encompassing the installation, repair, and maintenance of heating, ventilation, and air conditioning (HVAC) systems for residential, commercial, and industrial properties to control indoor climate.",
        "keywords": "air conditioning installation, furnace repair, ventilation systems, commercial Heating, Ventilation, and Air Conditioning service, residential heating and cooling, Heating, Ventilation, and Air Conditioning maintenance, thermostat installation"
    },
    {
        "label": "HVAC Inspections",
        "definition": "Professional assessment and evaluation of heating, ventilation, and air conditioning (HVAC) systems to check for proper operation, efficiency, safety, and compliance with regulations. Often done for real estate transactions or preventative maintenance.",
        "keywords": "heating system check, AC unit inspection, ventilation assessment, Heating, Ventilation, and Air Conditioning system evaluation, energy efficiency audit Heating, Ventilation, and Air Conditioning, preventative Heating, Ventilation, and Air Conditioning check, pre-purchase Heating, Ventilation, and Air Conditioning inspection"
    },
    {
        "label": "Air Duct Cleaning Services",
        "definition": "Services involving the cleaning of heating, ventilation, and air conditioning (HVAC) ductwork to remove dust, debris, allergens, and other contaminants, aiming to improve indoor air quality and system efficiency.",
        "keywords": "duct cleaning, ventilation system cleaning, Heating, Ventilation, and Air Conditioning ductwork service, indoor air quality improvement, allergen removal ducts, furnace duct cleaning, AC duct cleaning"
    },
    {
        "label": "Water Treatment Services",
        "definition": "Installation, maintenance, and repair of systems designed to improve water quality by removing impurities, contaminants, or undesirable minerals. This includes water softeners, filtration systems, and purification units for residential, commercial, or industrial use.",
        "keywords": "water filtration systems, water softener installation, reverse osmosis systems, water purification services, well water treatment, commercial water conditioning, industrial process water treatment"
    },
    {
        "label": "Residential Electrical Services",
        "definition": "Electrical services for private homes, including wiring installation, outlet and switch fitting, lighting installation, panel upgrades, and troubleshooting electrical issues.",
        "keywords": "home electrical repair, house wiring, lighting fixture installation, electrical panel upgrade, outlet installation, residential electrician, emergency electrical home"
    },
    {
        "label": "Commercial Electrical Services",
        "definition": "Electrical installation, maintenance, and repair services for businesses, offices, retail stores, and other commercial properties. This includes wiring for specialized equipment, commercial lighting, and higher voltage systems.",
        "keywords": "business electrician, office wiring, retail lighting installation, commercial electrical contractors, three-phase power, electrical safety inspections commercial, data cabling electrical"
    },
    {
        "label": "Alarm Installation Services",
        "definition": "Installation and setup of security alarm systems, fire alarm systems, or other warning systems in residential, commercial, or industrial properties to detect and alert to intrusions, fire, or other hazards.",
        "keywords": "security system setup, burglar alarm installation, fire alarm fitting, home security installation, commercial alarm systems, access control installation, surveillance system setup"
    },
    {
        "label": "Electric Line Construction",
        "definition": "Construction and maintenance of overhead and underground electrical power distribution and transmission lines. This involves setting poles, stringing conductors, and installing related equipment like transformers and switches.",
        "keywords": "power line installation, overhead electrical lines, underground power cables, utility line construction, electrical grid construction, transmission line services, distribution line maintenance"
    },
    {
        "label": "Cable Installation Services",
        "definition": "Installation of various types of cabling, including coaxial for television, twisted pair for telecommunications (phone/data), and fiber optic cables for high-speed internet and data transmission in residential or commercial settings.",
        "keywords": "data cabling, network cable installation, fiber optic installation, coaxial cable fitting, structured cabling solutions, telecommunication wiring, TV cable installation"
    },
    {
        "label": "Elevator Installation Services",
        "definition": "Specialized services for the assembly, installation, and commissioning of elevators, escalators, and moving walkways in new or existing buildings. This includes mechanical, electrical, and control system setup.",
        "keywords": "lift installation, escalator fitting, new elevator construction, elevator modernization, commercial elevator setup, residential elevator installation, accessibility lift services"
    },
    {
        "label": "Low-Rise Foundation Construction",
        "definition": "Construction of foundations (for example, basements, crawl spaces, slab-on-grade) for buildings typically up to a few stories high, such as single-family homes, townhouses, and small commercial structures.",
        "keywords": "residential foundation, shallow foundation, concrete slab, basement construction, crawl space foundation, small building foundation, footing installation"
    },
    {
        "label": "High-Rise Foundation Construction",
        "definition": "Specialized construction of deep and robust foundations (for example, piles, caissons, mat foundations) designed to support the significant loads of tall buildings, skyscrapers, and other large structures.",
        "keywords": "deep foundation systems, pile driving, caisson drilling, mat foundation, skyscraper foundation, geotechnical engineering construction, heavy load foundation"
    },
    {
        "label": "Precast Concrete Installation",
        "definition": "Services involving the assembly and erection of precast concrete components, such as wall panels, beams, columns, and floor slabs, that are manufactured off-site and transported for installation.",
        "keywords": "precast panel erection, concrete component assembly, modular concrete construction, structural precast installation, architectural precast fitting, off-site concrete solutions, accelerated building construction"
    },
    {
        "label": "Tilt-Up Concrete Services",
        "definition": "A construction method where large concrete wall panels are cast horizontally on-site and then tilted into their vertical position using cranes. This is common for warehouses and commercial buildings.",
        "keywords": "tilt-wall construction, concrete panel lifting, on-site casting, commercial building shells, warehouse construction method, crane-lifted panels, slab-cast walls"
    },
    {
        "label": "Masonry Construction Services",
        "definition": "Construction using materials like brick, stone, concrete blocks, and mortar to build walls, partitions, veneers, fireplaces, and other structures. This includes both structural and decorative masonry.",
        "keywords": "bricklaying, stonemasonry, blockwork, chimney construction, retaining wall building, veneer installation, tuckpointing services"
    },
    {
        "label": "Drywall Services",
        "definition": "Installation and finishing of drywall (gypsum board) panels to create interior walls and ceilings. This includes taping, mudding, and sanding to prepare surfaces for painting or wallpaper.",
        "keywords": "gypsum board installation, sheetrock hanging, drywall taping, wall finishing, ceiling installation, interior wall construction, plasterboard services"
    },
    {
        "label": "Tile Installation Services",
        "definition": "Services for installing ceramic, porcelain, stone, glass, or other types of tiles on floors, walls, backsplashes, showers, and other surfaces in residential and commercial properties.",
        "keywords": "floor tiling, wall tile setting, bathroom tile installation, kitchen backsplash tiling, ceramic tile contractor, porcelain tile fitting, mosaic tile work"
    },
    {
        "label": "Carpentry Services",
        "definition": "Skilled craftwork involving the cutting, shaping, and installation of wood and other building materials for framing, trim work, cabinetry, furniture, and other structural or decorative elements.",
        "keywords": "wood framing, finish carpentry, trim installation, custom cabinetry, deck building, home renovation carpentry, structural woodworking"
    },
    {
        "label": "Millwork Services",
        "definition": "Production and installation of custom or stock architectural wood elements, such as doors, windows, molding, paneling, cabinetry, and stairs, often requiring precision and decorative finishing.",
        "keywords": "architectural woodwork, custom molding, interior trim, bespoke cabinetry, stair construction, decorative wood paneling, fine woodworking"
    },
    {
        "label": "General Handyman Services",
        "definition": "Versatile services providing minor repairs, maintenance, and small installation tasks around residential or commercial properties. This can include plumbing, electrical, carpentry, and other odd jobs.",
        "keywords": "home repairs, property maintenance, odd job services, small project contractor, fix-it services, assembly services, minor installations"
    },
    {
        "label": "Insulation Services",
        "definition": "Installation of thermal or acoustic insulation materials in walls, attics, floors, and crawl spaces of buildings to improve energy efficiency, reduce noise, and enhance comfort.",
        "keywords": "attic insulation, wall insulation, spray foam insulation, fiberglass insulation, energy efficiency upgrades, soundproofing services, crawl space insulation"
    },
    {
        "label": "Painting Services",
        "definition": "Application of paint, stain, varnish, or other finishes to interior and exterior surfaces of buildings and structures for protection, aesthetics, and preservation.",
        "keywords": "interior painting, exterior house painting, commercial painting contractors, residential painters, surface preparation, decorative finishes, paint application"
    },
    {
        "label": "Spray Painting Services",
        "definition": "Application of paint or coatings using spray equipment, allowing for even and efficient coverage on various surfaces. Often used for large areas, complex shapes, or specialized finishes.",
        "keywords": "airless spraying, HVLP painting, industrial coatings application, automotive refinishing, large surface painting, uniform paint finish, specialty coatings"
    },
    {
        "label": "Flooring Installation Services",
        "definition": "Installation of various types of flooring materials, including hardwood, laminate, vinyl, carpet, tile, and resilient flooring, in residential and commercial settings.",
        "keywords": "hardwood floor fitting, carpet installation, vinyl plank flooring, tile flooring setup, laminate floor installation, commercial flooring solutions, subfloor preparation"
    },
    {
        "label": "Interior Design Services",
        "definition": "Professional services that plan, design, and furnish interior spaces to enhance their functionality, aesthetics, and safety, considering client needs, style preferences, and budget.",
        "keywords": "space planning, home decor consultation, commercial interior planning, furniture selection, color scheme design, lighting design, architectural interiors"
    },
    {
        "label": "Home Staging Services",
        "definition": "Preparing a residential property for sale by strategically arranging furniture, decor, and accessories to highlight its best features and appeal to potential buyers.",
        "keywords": "property styling, real estate staging, furniture arrangement for sale, house showcasing, buyer appeal enhancement, model home setup, pre-sale home preparation"
    },
    {
        "label": "Sheet Metal Services",
        "definition": "Fabrication and installation of components made from thin metal sheets, such as ductwork for HVAC systems, roofing flashing, gutters, and custom metal parts for various applications.",
        "keywords": "Heating, Ventilation, and Air Conditioning ductwork fabrication, metal roofing components, gutter installation, custom metal forming, sheet metal bending, welding sheet metal, precision metal work"
    },
    {
        "label": "Welding Services",
        "definition": "Joining metal parts using various welding processes (for example, MIG, TIG, Stick) to fabricate, repair, or modify metal structures, components, and equipment.",
        "keywords": "metal fabrication, custom welding, structural welding, pipe welding, aluminum welding, stainless steel welding, mobile welding repair"
    },
    {
        "label": "Structural Steel Erection",
        "definition": "The assembly and installation of steel frameworks for buildings, bridges, and other large structures. This involves lifting and bolting or welding steel beams, columns, and trusses.",
        "keywords": "steel framing construction, building steel assembly, bridge steel erection, heavy steel lifting, bolting steel structures, welding structural steel, industrial steelwork"
    },
    {
        "label": "Non-Structural Steel Fabrication",
        "definition": "Creation of steel components and assemblies that are not primary load-bearing elements of a structure, such as stairs, railings, platforms, miscellaneous supports, and ornamental metalwork.",
        "keywords": "miscellaneous metals, architectural steelwork, custom steel fabrication, handrail manufacturing, steel stair construction, ornamental ironwork, light gauge steel framing"
    },
    {
        "label": "Windows and Doors Installation",
        "definition": "Services specializing in the fitting and installation of new or replacement windows and doors in residential, commercial, or industrial buildings. This includes various materials like wood, vinyl, aluminum, and fiberglass.",
        "keywords": "window replacement, door fitting, new construction windows, residential door installation, commercial window contractors, energy-efficient windows, patio door installation"
    },
    {
        "label": "Well Drilling Services",
        "definition": "Services that involve drilling boreholes into the ground to access underground water sources (water wells), or for geothermal energy, oil and gas exploration, or environmental monitoring.",
        "keywords": "water well boring, geothermal drilling, oil well drilling, borehole construction, groundwater exploration, residential well drilling, commercial well services"
    },
    {
        "label": "Directional Drilling Services",
        "definition": "A trenchless method of installing underground pipes, conduits, or cables along a predetermined path by using steerable drilling equipment. This minimizes surface disruption and is used for utility installations under roads, rivers, or existing structures.",
        "keywords": "trenchless pipe installation, horizontal directional drilling (HDD), steerable boring, utility conduit installation, underground cable laying, river crossing drilling, minimal impact drilling"
    },
    {
        "label": "Infrastructure Excavation",
        "definition": "Large-scale excavation work specifically for public or private infrastructure projects, such as roads, bridges, pipelines, utility networks, and large foundations.",
        "keywords": "civil engineering excavation, utility trenching, roadbed preparation, bridge abutment digging, pipeline corridor excavation, mass excavation, site development earthwork"
    },
    {
        "label": "New Ground Excavation",
        "definition": "Excavation activities undertaken on previously undisturbed land or for entirely new construction projects, as opposed to excavation for repairs or modifications to existing sites.",
        "keywords": "greenfield site excavation, virgin land development, initial site clearing, bulk earthworks new projects, land grading for new construction, subdivision earthmoving, foundational digging new build"
    },
    {
        "label": "Residential Roofing Services",
        "definition": "Installation, repair, and replacement of roofing systems on private homes, using materials such as shingles, tiles, metal, or flat roofing membranes to protect against weather elements.",
        "keywords": "home roof repair, shingle installation, new roof residential, roof leak repair, residential re-roofing, gutter installation home, skylight installation"
    },
    {
        "label": "Roofing Services with Heat Application",
        "definition": "Roofing services that utilize heat for the application or sealing of roofing materials, such as torch-down modified bitumen for flat roofs, or heat-welded single-ply membranes (TPO, PVC).",
        "keywords": "torch-down roofing, modified bitumen application, heat-welded TPO, PVC roofing systems, flat roof heat application, commercial hot roofing, asphaltic roofing heat"
    },
    {
        "label": "Waterproofing Services",
        "definition": "Application of specialized materials and techniques to prevent water intrusion into buildings or structures. This includes basement waterproofing, foundation sealing, deck waterproofing, and exterior wall coatings.",
        "keywords": "basement sealing, foundation waterproofing, exterior wall coating, deck waterproofing, moisture barrier installation, sump pump waterproofing, concrete waterproofing"
    },
    {
        "label": "Septic System Services",
        "definition": "Installation, repair, maintenance, and pumping of on-site wastewater treatment systems (septic tanks and leach fields) for properties not connected to municipal sewer lines.",
        "keywords": "septic tank installation, leach field repair, septic pumping, septic system inspection, wastewater system maintenance, perc test, alternative septic systems"
    },
    {
        "label": "Building Cleaning Services",
        "definition": "Commercial or residential cleaning services for the interior and/or exterior of buildings. This can range from routine janitorial work to specialized cleaning like window washing or pressure washing.",
        "keywords": "janitorial services, office cleaning, commercial cleaning, residential cleaning, window washing, pressure washing building, post-construction cleanup"
    },
    {
        "label": "Fire Safety Equipment Services",
        "definition": "Supply, installation, inspection, testing, and maintenance of fire safety equipment, including fire extinguishers, fire suppression systems, fire alarms, emergency lighting, and sprinkler systems.",
        "keywords": "fire extinguisher servicing, fire alarm testing, sprinkler system inspection, emergency lighting maintenance, fire suppression system installation, life safety equipment, fire code compliance"
    },
    {
        "label": "Testing and Inspection Services",
        "definition": "Independent services providing evaluation, testing, and inspection of materials, products, systems, or structures to ensure quality, safety, compliance with standards, or to identify defects. This can span many industries (for example, construction, manufacturing, environmental).",
        "keywords": "quality control testing, compliance inspection, non-destructive testing (NDT), material testing, safety audits, building code inspection, environmental testing services"
    },
    {
        "label": "Single Family Residential Construction",
        "definition": "Construction of detached or semi-detached dwelling units designed for occupancy by one family. This includes custom homes, tract homes, and speculative new builds.",
        "keywords": "new home building, custom home construction, detached house building, residential property development, spec home construction, single-unit dwelling, house builders"
    },
    {
        "label": "Single Family Renovation Services",
        "definition": "Remodeling, additions, and significant alterations to existing single-family homes. This can include kitchen and bathroom remodels, room additions, whole-house makeovers, and structural repairs.",
        "keywords": "home remodeling, house additions, kitchen renovation, bathroom remodel, residential improvement, property refurbishment, custom home upgrades"
    },
    {
        "label": "Multi-Family Construction Services",
        "definition": "Construction of residential buildings designed to house multiple separate families or dwelling units, such as apartment buildings, condominiums, townhouses, and duplexes.",
        "keywords": "apartment building construction, condo development, townhouse projects, multi-unit residential, low-rise apartment building, high-rise residential construction, mixed-use residential"
    },
    {
        "label": "Apartment Renovation Services",
        "definition": "Remodeling and updating of individual apartment units or entire apartment buildings, including kitchen and bath upgrades, flooring, painting, and common area improvements.",
        "keywords": "apartment unit remodel, building-wide renovation, multi-family property upgrades, tenant improvement apartments, common area refurbishment, condo renovation, investment property rehab"
    },
    {
        "label": "Restoration Services",
        "definition": "Repair and rebuilding of properties damaged by events such as fire, water, storms, or mold. Services aim to return the property to its pre-loss condition or better.",
        "keywords": "fire damage repair, water damage cleanup, storm damage restoration, mold remediation, property disaster recovery, emergency board-up, reconstruction services"
    },
    {
        "label": "Mobile Home Construction Services",
        "definition": "Factory construction of prefabricated homes that are transportable in one or more sections and designed for long-term residential use when connected to utilities. This also includes on-site setup and installation.",
        "keywords": "manufactured home building, prefabricated housing, modular home construction (if mobile), park model homes, mobile home setup, factory-built homes, transportable dwellings"
    },
    {
        "label": "Commercial Construction Services",
        "definition": "Construction of buildings and structures for business, retail, office, industrial, or institutional use. This includes new builds, tenant improvements, and expansions.",
        "keywords": "office building construction, retail space build-out, industrial facility construction, warehouse building, new commercial development, tenant fit-out, institutional building"
    },
    {
        "label": "Commercial Renovation Services",
        "definition": "Remodeling, updating, and reconfiguring existing commercial properties to meet new business needs, improve functionality, or modernize aesthetics. This includes office remodels, retail store makeovers, and facility upgrades.",
        "keywords": "office remodel, retail store renovation, business property upgrades, tenant improvement commercial, commercial fit-out, industrial facility refurbishment, adaptive reuse projects"
    },
    {
        "label": "Swimming Pool Installation Services",
        "definition": "Services specializing in the design and construction of in-ground or above-ground swimming pools for residential or commercial properties. This includes excavation, plumbing, and finishing.",
        "keywords": "pool building, in-ground pool construction, above-ground pool setup, residential pool installation, commercial pool contractors, pool design, spa installation"
    },
    {
        "label": "Swimming Pool Maintenance Services",
        "definition": "Regular upkeep and cleaning services for swimming pools, including water testing and chemical balancing, debris removal, filter cleaning, and equipment checks to ensure safe and clean water.",
        "keywords": "pool cleaning, pool water treatment, chemical balancing, filter maintenance, weekly pool service, pool opening/closing, spa maintenance"
    },
    {
        "label": "Vacant Building Management",
        "definition": "Services for overseeing and maintaining unoccupied commercial or residential buildings, including security, routine inspections, basic repairs, and ensuring compliance with local ordinances to prevent deterioration or vandalism.",
        "keywords": "empty property care, unoccupied building security, property inspections vacant, caretaker services, preserving vacant assets, remote property monitoring, preventative maintenance vacant"
    },
    {
        "label": "Vacant Land Services",
        "definition": "Management and maintenance of undeveloped or unused parcels of land. This can include mowing, brush clearing, debris removal, security patrols, and ensuring compliance with land use regulations.",
        "keywords": "land clearing, lot maintenance, brush hogging, undeveloped property care, groundskeeping vacant land, vegetation control, land stewardship"
    },
    {
        "label": "Meat Processing Services",
        "definition": "Operations involving the slaughtering of livestock and poultry, and the subsequent cutting, processing, packaging, and sometimes curing or smoking of meat products for wholesale or retail sale.",
        "keywords": "abattoir services, butchery, meat packing, poultry processing, custom meat cutting, sausage making, slaughterhouse operations"
    },
    {
        "label": "Seafood Processing Services",
        "definition": "Transforming raw fish and shellfish into various marketable products. This includes heading, gutting, filleting, shucking, freezing, canning, smoking, or packaging seafood.",
        "keywords": "fish filleting, shellfish shucking, seafood packing, fish processing plant, frozen seafood production, canned fish, value-added seafood"
    },
    {
        "label": "Dairy Production Services",
        "definition": "Operations focused on the collection of milk from dairy animals (for example, cows, goats) and its processing into various dairy products like pasteurized milk, cheese, yogurt, butter, and ice cream.",
        "keywords": "milk pasteurization, cheese making, yogurt production, butter manufacturing, ice cream production, dairy farm processing, raw milk handling"
    },
    {
        "label": "Frozen Food Processing",
        "definition": "Manufacturing processes that involve preparing, cooking (if applicable), and then rapidly freezing food items for extended shelf life and distribution. This includes vegetables, fruits, meats, ready-to-eat meals, and desserts.",
        "keywords": "flash freezing, ready-meal production frozen, frozen vegetable processing, frozen fruit packing, blast chilling, IQF (Individually Quick Frozen) products, frozen dessert manufacturing"
    },
    {
        "label": "Ice Production Services",
        "definition": "Manufacturing and packaging of ice in various forms (for example, cubed, crushed, block) for commercial sale to restaurants, retailers, events, or industrial use.",
        "keywords": "commercial ice making, packaged ice manufacturing, ice delivery services, block ice production, dry ice manufacturing (if applicable), industrial ice supply, bagged ice"
    },
    {
        "label": "Canning Services",
        "definition": "Preserving food by sealing it in airtight containers (cans or jars) and then heat-treating it to destroy microorganisms and enzymes. This is used for fruits, vegetables, meats, and prepared foods.",
        "keywords": "food preservation canning, fruit canning, vegetable canning, commercial cannery, retort processing, jarred food production, shelf-stable food processing"
    },
    {
        "label": "Animal Feed Manufacturing",
        "definition": "Production of balanced nutritional food products for livestock, poultry, aquaculture, and other farm animals. This involves sourcing ingredients, milling, mixing, and often pelletizing or extruding.",
        "keywords": "livestock feed production, poultry feed mill, aquaculture feed, custom feed formulation, cattle feed manufacturing, swine feed, pet food production (can overlap)"
    },
    {
        "label": "Pet Food Manufacturing",
        "definition": "Production of commercially prepared food for domestic animals like dogs, cats, birds, and small mammals. This includes dry kibble, wet food, treats, and specialized dietary formulas.",
        "keywords": "dog food production, cat food manufacturing, dry pet food kibble, wet pet food canning, pet treat manufacturing, animal nutrition pet, premium pet food"
    },
    {
        "label": "Grain Processing Services",
        "definition": "Transforming raw grains (for example, wheat, corn, rice, oats) into various food ingredients or products through processes like milling (flour), cleaning, sorting, rolling, or malting.",
        "keywords": "flour milling, corn milling, rice processing, oat rolling, grain cleaning, seed sorting, malt production"
    },
    {
        "label": "Coffee Processing Services",
        "definition": "Transforming raw coffee cherries into green coffee beans ready for roasting. This includes processes like pulping, fermenting, washing, drying, hulling, sorting, and grading.",
        "keywords": "coffee bean pulping, coffee cherry drying, green coffee bean production, coffee hulling, coffee grading, wet processing coffee, dry processing coffee"
    },
    {
        "label": "Seed Processing Services",
        "definition": "Cleaning, treating, and packaging of agricultural or horticultural seeds to improve their quality, viability, and plantability for farmers or gardeners.",
        "keywords": "seed cleaning, seed treating, seed coating, seed packaging, agricultural seed conditioning, certified seed production, seed quality enhancement"
    },
    {
        "label": "Bakery Production Services",
        "definition": "Commercial production of baked goods such as bread, cakes, pastries, cookies, and pies. This can range from small artisanal bakeries to large-scale industrial operations.",
        "keywords": "bread baking, cake manufacturing, pastry production, commercial bakery, wholesale baking, artisanal bread, cookie production"
    },
    {
        "label": "Confectionery Manufacturing",
        "definition": "Production of sweet food items like candies, chocolates, chewing gum, and other sugar-based treats. This involves mixing, cooking, molding, and packaging.",
        "keywords": "candy making, chocolate production, sugar confectionery, gum manufacturing, sweet treats, artisanal chocolates, bulk candy production"
    },
    {
        "label": "Distilling Services",
        "definition": "The process of purifying a liquid by successive evaporation and condensation, primarily used to produce alcoholic spirits like whiskey, vodka, gin, rum, and brandy from fermented mashes.",
        "keywords": "spirit production, alcohol distillation, whiskey distillery, vodka manufacturing, gin making, craft distillery, micro-distillery"
    },
    {
        "label": "Brewery Operations",
        "definition": "The business of brewing and selling beer. This encompasses the entire process from sourcing ingredients (malt, hops, yeast, water) to mashing, fermenting, conditioning, and packaging beer.",
        "keywords": "beer brewing, craft brewery, microbrewery, ale production, lager manufacturing, beer fermentation, brewpub operations"
    },
    {
        "label": "Non-Alcoholic Beverage Manufacturing",
        "definition": "Production of beverages that do not contain alcohol, such as soft drinks, juices, bottled water, teas, coffees (ready-to-drink), and sports drinks.",
        "keywords": "soft drink production, juice bottling, bottled water plant, tea manufacturing, ready-to-drink coffee, sports drink production, carbonated beverage manufacturing"
    },
    {
        "label": "Oil and Fat Manufacturing",
        "definition": "Industrial processing of plant or animal raw materials to extract and refine oils and fats for human consumption, animal feed, or industrial applications. This includes crushing, pressing, solvent extraction, and refining.",
        "keywords": "vegetable oil production, animal fat rendering, edible oil refining, seed oil extraction, shortening manufacturing, margarine production, industrial lubricants"
    },
    {
        "label": "Rendering Services",
        "definition": "Processing of animal by-products (for example, fat, bone, offal) into stable, usable materials such as tallow, grease, and protein meals. These rendered products are used in animal feed, soap, and other industrial applications.",
        "keywords": "animal by-product processing, tallow production, protein meal manufacturing, fat recycling, inedible rendering, meat industry by-products, grease processing"
    },
    {
        "label": "Textile Manufacturing Services",
        "definition": "The process of converting natural or synthetic fibers into yarn, and then into woven, knitted, or non-woven fabrics. This can also include dyeing, printing, and finishing processes.",
        "keywords": "fabric production, yarn spinning, weaving mills, knitting mills, non-woven fabric manufacturing, textile dyeing, fabric finishing"
    },
    {
        "label": "Carpet Manufacturing Services",
        "definition": "Production of carpets and rugs through processes like tufting, weaving, or needle-punching. This includes creating various pile types, patterns, and backing materials for residential or commercial use.",
        "keywords": "rug production, tufted carpet manufacturing, woven carpets, commercial carpeting, residential carpet production, carpet backing, floor covering manufacturing"
    },
    {
        "label": "Apparel Manufacturing",
        "definition": "The mass production of clothing items, including cutting fabric, sewing garments, and finishing details. This covers a wide range of clothing for men, women, and children.",
        "keywords": "garment production, clothing factory, sewing operations, fashion manufacturing, cut and sew, apparel assembly, clothing line production"
    },
    {
        "label": "Accessory Manufacturing",
        "definition": "Production of fashion and functional accessories such as handbags, belts, hats, scarves, gloves, wallets, and small leather goods.",
        "keywords": "handbag production, belt making, hat manufacturing, fashion accessory creation, leather goods production, scarf weaving, small goods assembly"
    },
    {
        "label": "Children\'s Clothing Manufacturing",
        "definition": "Specialized production of apparel designed for infants, toddlers, and children, often with a focus on safety, comfort, durability, and age-appropriate designs.",
        "keywords": "kids\' apparel production, infant wear manufacturing, toddler clothing factory, children\'s garment sewing, baby clothes production, youth fashion, school uniform manufacturing"
    },
    {
        "label": "Window Treatment Manufacturing",
        "definition": "Production of items used to cover windows for privacy, light control, or decoration, such as blinds, shades, curtains, drapes, and shutters.",
        "keywords": "blind manufacturing, shade production, curtain making, drapery fabrication, custom window coverings, shutter assembly, interior window solutions"
    },
    {
        "label": "Canvas Manufacturing",
        "definition": "Production of heavy-duty plain-woven fabric, typically made from cotton or linen, used for sails, tents, awnings, artist canvases, and other durable applications.",
        "keywords": "heavy fabric production, industrial canvas weaving, cotton duck manufacturing, artist canvas making, tent fabric production, awning material, durable textile weaving"
    },
    {
        "label": "Rope Production Services",
        "definition": "Manufacturing of ropes, cords, and twines by twisting or braiding natural or synthetic fibers. This includes ropes for marine, industrial, climbing, and general utility purposes.",
        "keywords": "cordage manufacturing, synthetic rope production, natural fiber ropes, marine rope making, climbing rope fabrication, industrial cord production, twine twisting"
    },
    {
        "label": "Tent Manufacturing Services",
        "definition": "Design and production of portable shelters made from fabric (often canvas or nylon) supported by poles or frames. This includes camping tents, event tents, and industrial or military tents.",
        "keywords": "shelter fabrication, camping tent production, event marquee manufacturing, industrial tent making, outdoor gear manufacturing, canvas tent sewing, portable structure production"
    },
    {
        "label": "Wood Product Manufacturing",
        "definition": "Transforming raw lumber or wood materials into a wide variety of finished or semi-finished goods, excluding furniture and cabinetry which are often distinct categories. This can include engineered wood products, millwork, or specialized wood components.",
        "keywords": "lumber processing (secondary), engineered wood production, wood components manufacturing, building materials wood, specialized wood items, wood turning, architectural wood products"
    },
    {
        "label": "Window and Door Manufacturing",
        "definition": "Factory production of complete window and door units, including frames (wood, vinyl, aluminum, fiberglass), sashes, glazing, and hardware, for residential or commercial buildings.",
        "keywords": "window frame assembly, door fabrication, residential window production, commercial door manufacturing, energy-efficient window making, custom window and door, fenestration products"
    },
    {
        "label": "Cabinetry Manufacturing",
        "definition": "Design and production of cabinets for kitchens, bathrooms, offices, and other storage applications. This includes stock, semi-custom, and custom cabinet making from various materials like wood, laminates, or MDF.",
        "keywords": "kitchen cabinet production, bathroom vanity manufacturing, custom cabinet making, built-in furniture, storage solutions, wood cabinet fabrication, modular cabinetry"
    },
    {
        "label": "Trim Manufacturing",
        "definition": "Production of decorative and functional wood or composite moldings and trim pieces used in construction and finishing, such as baseboards, crown molding, casing, and chair rails.",
        "keywords": "wood molding production, architectural trim, baseboard manufacturing, crown molding fabrication, interior trim making, decorative millwork, casing production"
    },
    {
        "label": "Pallet Manufacturing",
        "definition": "Production of wooden, plastic, or metal pallets used for shipping and storing goods. This involves cutting materials to size and assembling them into a flat transport structure.",
        "keywords": "wooden pallet production, shipping pallet assembly, industrial pallets, plastic pallet molding, custom pallet design, logistics support manufacturing, material handling pallets"
    },
    {
        "label": "Furniture Manufacturing",
        "definition": "Design and production of movable objects intended to support various human activities (for example, seating, sleeping, storage) or to hold objects at a convenient height. This includes residential, office, and institutional furniture.",
        "keywords": "home furniture production, office furniture manufacturing, commercial seating, wood furniture making, upholstered furniture, metal furniture fabrication, case goods production"
    },
    {
        "label": "Bedding Manufacturing",
        "definition": "Production of items used for beds, primarily mattresses and box springs/foundations. This can also include pillows, comforters, and other bed coverings.",
        "keywords": "mattress production, box spring assembly, sleep products manufacturing, pillow making, comforter production, bed foundation, sleep system fabrication"
    },
    {
        "label": "Paper Production Services",
        "definition": "Manufacturing of paper from wood pulp, recycled paper, or other cellulose fibers. This involves processes like pulping, forming sheets, pressing, drying, and finishing into rolls or sheets.",
        "keywords": "pulp and paper mill, paper making, newsprint production, fine paper manufacturing, recycled paper production, specialty paper, paperboard manufacturing"
    },
    {
        "label": "Stationery Manufacturing",
        "definition": "Production of writing materials and office supplies, such as notebooks, envelopes, writing paper, greeting cards, and other paper-based products used for correspondence or record-keeping.",
        "keywords": "notebook production, envelope making, greeting card printing, writing pad manufacturing, office paper products, personal stationery, school supplies paper"
    },
    {
        "label": "Publishing Services",
        "definition": "Businesses involved in the selection, preparation, production, marketing, and distribution of printed or digital content, such as books, magazines, newspapers, journals, and online publications.",
        "keywords": "book publishing, magazine production, newspaper printing, digital publishing, content distribution, manuscript editing, e-book creation"
    },
    {
        "label": "Printing Services",
        "definition": "Commercial services offering reproduction of text and images, typically with ink on paper or other materials, using various printing technologies like offset, digital, or screen printing.",
        "keywords": "commercial printing, digital printing, offset printing, large format printing, brochure printing, business card production, document reproduction"
    },
    {
        "label": "Gas Manufacturing Services",
        "definition": "Industrial production of various gases, including industrial gases (oxygen, nitrogen, argon), specialty gases, and fuel gases, through processes like air separation, chemical reactions, or purification.",
        "keywords": "industrial gas production, specialty gas manufacturing, air separation unit, hydrogen production, oxygen generation, compressed gas supply, liquefied natural gas (LNG)"
    },
    {
        "label": "Ink Production Services",
        "definition": "Manufacturing of inks used for printing, writing, or other applications. This involves formulating pigments, dyes, solvents, and other additives to achieve desired color, viscosity, and performance characteristics.",
        "keywords": "printing ink manufacturing, writing ink production, specialty inks, pigment formulation, solvent-based inks, water-based inks, UV curable inks"
    },
    {
        "label": "Plastic Manufacturing",
        "definition": "Transformation of raw plastic resins (polymers) into finished or semi-finished products using processes like injection molding, blow molding, extrusion, thermoforming, or rotational molding.",
        "keywords": "injection molding, plastic extrusion, blow molding services, plastic product fabrication, polymer processing, thermoforming, custom plastic parts"
    },
    {
        "label": "Cosmetic Manufacturing",
        "definition": "Production of beauty and personal care products, including makeup, skincare, haircare, and fragrances. This involves formulation, mixing, filling, and packaging.",
        "keywords": "beauty product production, skincare formulation, makeup manufacturing, haircare products, fragrance creation, private label cosmetics, personal care items"
    },
    {
        "label": "Soap Production Services",
        "definition": "Manufacturing of cleaning agents made from fats/oils and alkali, in bar, liquid, or powder form for personal hygiene or general cleaning purposes.",
        "keywords": "bar soap making, liquid soap production, detergent manufacturing (if similar process), handcrafted soap, industrial soap production, cosmetic soap, cleaning agent manufacturing"
    },
    {
        "label": "Chemical Manufacturing",
        "definition": "Large-scale production of chemicals through various chemical processes and reactions. This includes bulk chemicals, specialty chemicals, petrochemicals, pharmaceuticals, and agrochemicals.",
        "keywords": "industrial chemical production, specialty chemicals, bulk chemical manufacturing, organic synthesis, inorganic chemical production, petrochemical plant, fine chemicals"
    },
    {
        "label": "Asphalt Production Services",
        "definition": "Manufacturing of asphalt concrete (asphalt mix) by combining aggregates (sand, gravel, crushed stone) with a bitumen binder, typically in an asphalt mixing plant, for use in paving roads, driveways, and parking lots.",
        "keywords": "asphalt mixing plant, hot mix asphalt (HMA), paving material production, bitumen binder, road surfacing material, cold mix asphalt, aggregate coating"
    },
    {
        "label": "Rubber Manufacturing",
        "definition": "Processing of natural or synthetic rubber to create a wide range of products, including tires, hoses, seals, gaskets, and other molded or extruded rubber goods.",
        "keywords": "rubber molding, tire production, rubber extrusion, synthetic rubber products, natural rubber processing, industrial rubber goods, custom rubber parts"
    },
    {
        "label": "Plastic Signage Production",
        "definition": "Manufacturing of signs primarily made from plastic materials, such as acrylic, PVC, or polycarbonate. This includes cutting, forming, printing, and assembling plastic signs for various applications.",
        "keywords": "acrylic sign making, PVC signage, plastic lettering, illuminated plastic signs, retail signage plastic, directional signs plastic, custom plastic displays"
    },
    {
        "label": "Media Production Services",
        "definition": "Creation of content for various media platforms, including film, television, radio, video games, and online digital media. This encompasses pre-production, production (filming/recording), and post-production (editing).",
        "keywords": "film production, video creation, television programming, audio recording services, digital content creation, animation services, post-production editing"
    },
    {
        "label": "Software Manufacturing",
        "definition": "While \"manufacturing\" is less common for software, this refers to the development, duplication, packaging, and distribution of software products, whether on physical media or via digital download. Often used for mass-market or enterprise software.",
        "keywords": "software development (product), software duplication, software packaging, licensed software distribution, enterprise software solutions, shrink-wrapped software, digital software delivery"
    },
    {
        "label": "Food Processing Services",
        "definition": "Transforming raw agricultural products into food, or one form of food into other forms. This includes a wide range of activities like cleaning, cutting, cooking, mixing, preserving, and packaging food items.",
        "keywords": "food manufacturing, packaged food production, ingredient processing, food preservation techniques, ready-to-eat meal preparation, commercial kitchen operations, value-added food products"
    },
    {
        "label": "Pharmaceutical Manufacturing",
        "definition": "Large-scale production of medicinal drugs and pharmaceutical products. This involves precise formulation, synthesis or extraction of active ingredients, quality control, and packaging in various dosage forms.",
        "keywords": "drug production, medication manufacturing, active pharmaceutical ingredient (API) synthesis, sterile drug production, tablet manufacturing, vaccine production, GMP compliance"
    },
    {
        "label": "Laboratory Services",
        "definition": "Facilities providing analytical testing, research, and diagnostic services across various fields such as medical, environmental, industrial, and scientific research.",
        "keywords": "analytical testing lab, diagnostic testing, research laboratory, environmental testing, clinical lab services, materials testing, scientific analysis"
    },
    {
        "label": "Waste Management Services",
        "definition": "Collection, transportation, processing, and disposal or recycling of various types of waste materials from residential, commercial, and industrial sources.",
        "keywords": "trash collection, refuse disposal, recycling services, landfill operations, hazardous waste disposal, dumpster rental, sanitation services"
    },
    {
        "label": "Recycling Services",
        "definition": "Collection and processing of used materials (e.g., paper, plastic, glass, metal) to convert them into new raw materials or products, diverting them from landfills.",
        "keywords": "material recovery facility (MRF), paper recycling, plastic reprocessing, glass recycling, metal scrap collection, e-waste recycling, circular economy solutions"
    },
    {
        "label": "Environmental Consulting",
        "definition": "Providing expert advice and solutions to businesses or government agencies on environmental issues, such as regulatory compliance, pollution control, site remediation, impact assessments, and sustainability.",
        "keywords": "environmental compliance, pollution prevention, site assessment, environmental impact studies, sustainability consulting, remediation services, ecological consulting"
    },
    {
        "label": "Property Management Services",
        "definition": "Overseeing and managing real estate properties on behalf of owners. This includes tenant screening, rent collection, property maintenance, and financial reporting for residential or commercial properties.",
        "keywords": "rental property management, commercial real estate services, tenant relations, lease administration, building maintenance oversight, investment property management, real estate asset management"
    },
    {
        "label": "Real Estate Services",
        "definition": "Services related to the buying, selling, leasing, and appraising of real property (land and buildings). This includes real estate brokerage, agent services, and property valuation.",
        "keywords": "real estate brokerage, property sales, home buying/selling, commercial leasing, property appraisal, real estate agent, land sales"
    },
    {
        "label": "Insurance Services",
        "definition": "Businesses providing risk management solutions through insurance policies. This includes underwriting, selling, and administering insurance coverage for various risks (life, health, property, casualty).",
        "keywords": "insurance brokerage, underwriting services, policy administration, claims processing, risk management solutions, life insurance, property and casualty insurance"
    },
    {
        "label": "Financial Services",
        "definition": "Services provided by the finance industry, including banking, investment management, insurance, financial planning, lending, and accounting.",
        "keywords": "banking services, investment advice, financial planning, wealth management, loan origination, accounting services, stock brokerage"
    },
    {
        "label": "Legal Services",
        "definition": "Professional advice and representation provided by lawyers and law firms on legal matters, including litigation, corporate law, real estate law, criminal defense, and family law.",
        "keywords": "law firm, attorney services, litigation support, corporate counsel, legal consultation, contract law, legal representation"
    },
    {
        "label": "Consulting Services",
        "definition": "Providing expert advice and solutions to organizations or individuals in a particular area of expertise, such as management, IT, strategy, operations, or human resources.",
        "keywords": "business consulting, IT consulting, management advice, strategy consulting, operational improvement, expert advisory, professional guidance"
    },
    {
        "label": "Marketing Services",
        "definition": "Services that help businesses promote and sell their products or services. This includes market research, advertising, public relations, digital marketing, content creation, and branding.",
        "keywords": "advertising agency, digital marketing, public relations, market research, branding services, content creation, SEO services"
    },
    {
        "label": "Human Resources Services",
        "definition": "Services related to managing an organization\'s workforce, including recruitment, staffing, payroll, benefits administration, employee relations, training, and HR consulting.",
        "keywords": "recruitment agency, payroll processing, benefits management, HR consulting, talent acquisition, employee training, outsourced HR"
    },
    {
        "label": "Management Consulting",
        "definition": "Providing advisory services to organizations to help improve their performance, primarily through the analysis of existing organizational problems and the development of plans for improvement.",
        "keywords": "business strategy, organizational development, process improvement, operational excellence, change management, performance consulting, executive coaching"
    },
    {
        "label": "Business Development Services",
        "definition": "Services aimed at helping organizations grow by identifying new market opportunities, building client relationships, developing strategic partnerships, and increasing revenue.",
        "keywords": "sales strategy, market expansion, lead generation, client acquisition, strategic partnerships, revenue growth, new business opportunities"
    },
    {
        "label": "Project Management Services",
        "definition": "Planning, organizing, securing, managing, leading, and controlling resources to achieve specific goals within a defined scope, time, and budget for projects.",
        "keywords": "project planning, resource allocation, risk management project, scope management, project execution, timeline control, stakeholder management"
    },
    {
        "label": "Technology Consulting",
        "definition": "Providing expert advice and services to organizations on how to best use information technology (IT) to achieve their business objectives. This includes strategy, implementation, and management of IT systems.",
        "keywords": "IT consulting, technology advisory, systems integration, digital transformation, IT strategy, cloud consulting, cybersecurity consulting"
    },
    {
        "label": "Software Development Services",
        "definition": "Designing, creating, deploying, and maintaining software applications for specific client needs or for commercial distribution. This includes custom software, web applications, and mobile apps.",
        "keywords": "custom software creation, application development, web app development, mobile app development, programming services, software engineering, coding services"
    },
    {
        "label": "Data Analysis Services",
        "definition": "Collecting, processing, inspecting, cleaning, transforming, and modeling data to discover useful information, inform conclusions, and support decision-making for businesses.",
        "keywords": "data analytics, business intelligence, statistical analysis, data modeling, reporting services, data interpretation, big data analysis"
    },
    {
        "label": "Market Research Services",
        "definition": "Gathering, analyzing, and interpreting information about a market, product or service to be offered for sale in that market, and about the past, present, and potential customers for the product or service.",
        "keywords": "consumer research, industry analysis, competitor analysis, survey design, focus groups, market segmentation, data collection market"
    },
    {
        "label": "Strategic Planning Services",
        "definition": "Assisting organizations in defining their strategy or direction, and making decisions on allocating resources to pursue this strategy, including capital and people.",
        "keywords": "business strategy formulation, long-term planning, corporate strategy, goal setting, competitive positioning, market entry strategy, organizational planning"
    },
    {
        "label": "Training Services",
        "definition": "Providing instruction and skill development programs to individuals or groups within organizations to improve job performance, enhance knowledge, or ensure compliance.",
        "keywords": "corporate training, employee development, skill workshops, professional development, compliance training, onboarding programs, instructional design"
    },
    {
        "label": "Public Relations Services",
        "definition": "Managing the spread of information between an individual or an organization (such as a business, government agency, or a nonprofit organization) and the public to shape public perception.",
        "keywords": "media relations, press release distribution, corporate communications, reputation management, crisis communication, publicity services, influencer outreach"
    },
    {
        "label": "Event Planning Services",
        "definition": "Coordinating and managing all aspects of professional or social events, including conferences, meetings, weddings, parties, and conventions, from conception to completion.",
        "keywords": "meeting planning, conference management, wedding coordination, corporate events, party planning, logistics management events, venue selection"
    },
    {
        "label": "Catering Services",
        "definition": "Providing food and beverage services for events, functions, or institutions at remote sites or locations such as hotels, hospitals, pubs, aircraft, cruise ships, parks, filming locations, or private homes.",
        "keywords": "event catering, corporate catering, wedding food service, mobile catering, food delivery events, banquet services, private party catering"
    },
    {
        "label": "Travel Services",
        "definition": "Assisting individuals or groups in planning and booking travel arrangements, including flights, accommodations, tours, transportation, and travel insurance.",
        "keywords": "travel agency, flight booking, hotel reservations, tour packages, corporate travel management, vacation planning, cruise booking"
    },
    {
        "label": "Digital Marketing Services",
        "definition": "Utilizing online channels like search engines, social media, email, and websites to connect with current and prospective customers and promote products or services.",
        "keywords": "online advertising, Search Engine Optimization/Search Engine Marketing, social media marketing, email marketing campaigns, content marketing online, PPC advertising, web analytics"
    },
    {
        "label": "E-Commerce Services",
        "definition": "Services supporting online sales transactions, including website development, payment processing integration, online store management, order fulfillment, and digital marketing for online retailers.",
        "keywords": "online store development, shopping cart solutions, payment gateway integration, online retail management, digital storefront, Business-to-Consumer e-commerce, Business-to-Business e-commerce platforms"
    },
    {
        "label": "Online Marketing Services",
        "definition": "A broad term encompassing various strategies used to market products or services online, including SEO, SEM, social media marketing, content marketing, email marketing, and affiliate marketing. (Similar to Digital Marketing Services but sometimes used more broadly).",
        "keywords": "internet marketing, web advertising, online promotion, digital strategy, lead generation online, affiliate marketing, search engine marketing (SEM)"
    },
    {
        "label": "Content Creation Services",
        "definition": "Producing various forms of material for online or offline channels, such as blog posts, articles, website copy, videos, infographics, podcasts, and social media updates, often for marketing or informational purposes.",
        "keywords": "blog writing, copywriting services, video production marketing, infographic design, podcast creation, social media content, website content development"
    },
    {
        "label": "Website Development Services",
        "definition": "Designing, building, and maintaining websites for individuals or organizations. This includes front-end (user interface) and back-end (server-side logic and database) development.",
        "keywords": "web design, web programming, front-end development, back-end development, custom website creation, CMS development (WordPress, Drupal), website maintenance"
    },
    {
        "label": "SEO Services",
        "definition": "Search Engine Optimization services aimed at improving a website\'s visibility and ranking in search engine results pages (SERPs) through on-page optimization, link building, and technical SEO.",
        "keywords": "search engine ranking, keyword research, on-page optimization, link building strategy, technical SEO audit, local SEO, organic traffic growth"
    },
    {
        "label": "Social Media Services",
        "definition": "Managing and executing marketing strategies on social media platforms like Facebook, Instagram, Twitter, LinkedIn, and others. This includes content creation, posting schedules, community engagement, and advertising campaigns.",
        "keywords": "social media management, community management, Facebook advertising, Instagram marketing, content scheduling social, social media campaigns, influencer marketing"
    },
    {
        "label": "Branding Services",
        "definition": "Developing and managing a brand\'s identity, including logo design, brand messaging, visual style guides, and overall brand strategy to shape public perception and differentiate from competitors.",
        "keywords": "brand identity design, logo creation, brand strategy, visual identity, corporate branding, brand messaging, style guide development"
    },
    {
        "label": "Graphic Design Services",
        "definition": "Creating visual concepts, using computer software or by hand, to communicate ideas that inspire, inform, and captivate consumers. This includes designs for logos, brochures, advertisements, websites, and packaging.",
        "keywords": "visual communication, logo design, marketing material design, publication layout, illustration services, web graphics, print design"
    },
    {
        "label": "Advertising Services",
        "definition": "Creating, planning, and executing advertising campaigns across various media channels (print, broadcast, online, outdoor) to promote products, services, or brands to a target audience.",
        "keywords": "ad campaign creation, media planning, media buying, creative development advertising, print advertising, broadcast commercials, online ad placement"
    },
    {
        "label": "Corporate Training Services",
        "definition": "Designing and delivering training programs specifically for businesses and their employees, covering topics like leadership, sales, technical skills, compliance, and soft skills.",
        "keywords": "employee education, professional development workshops, leadership training, sales training programs, skills development corporate, compliance education, soft skills training"
    },
    {
        "label": "Health and Safety Consulting",
        "definition": "Providing expert advice and services to organizations to help them identify hazards, manage risks, and comply with occupational health and safety (OHS) regulations to ensure a safe workplace.",
        "keywords": "Occupational Health and Safety consulting, workplace safety audits, risk management safety, safety program development, regulatory compliance Occupational Health and Safety, hazard assessment, ergonomic consulting"
    },
    {
        "label": "Food Safety Services",
        "definition": "Services aimed at ensuring food products are safe for consumption, including consulting, auditing, training, and testing related to food handling, preparation, storage, and HACCP compliance.",
        "keywords": "Hazard Analysis and Critical Control Points consulting, food handling training, kitchen safety audits, food hygiene services, foodborne illness prevention, food quality testing, regulatory compliance food"
    },
    {
        "label": "Quality Assurance Services",
        "definition": "Implementing and managing processes and systems to ensure that products or services consistently meet specified quality standards and customer expectations. This includes testing, inspection, and process auditing.",
        "keywords": "QA testing, quality control systems, process auditing, standards compliance, product inspection, ISO 9001 consulting, quality management systems (QMS)"
    },
    {
        "label": "Compliance Services",
        "definition": "Assisting organizations in adhering to relevant laws, regulations, standards, and internal policies across various domains like finance, environmental, safety, data privacy, and industry-specific rules.",
        "keywords": "regulatory compliance consulting, policy adherence, corporate governance, legal compliance audit, risk mitigation compliance, industry standards compliance, data privacy compliance (General Data Protection Regulation, California Consumer Privacy Act)"
    },
    {
        "label": "Environmental Health Services",
        "definition": "Services focused on identifying, assessing, and controlling environmental factors that can potentially affect human health, such as air and water quality, hazardous materials, sanitation, and vector control.",
        "keywords": "public health environmental, air quality monitoring, water safety testing, hazardous material management, sanitation inspection, vector control programs, industrial hygiene"
    },
    {
        "label": "Risk Assessment Services",
        "definition": "Identifying potential hazards or threats (such as financial, operational, safety, or security), analyzing the likelihood and potential impact, and recommending mitigation strategies for organizations.",
        "keywords": "threat analysis, vulnerability assessment, impact analysis, risk mitigation planning, security risk assessment, financial risk modeling, operational risk management"
    },
    {
        "label": "Crisis Management Services",
        "definition": "Helping organizations prepare for, respond to, and recover from major disruptive events (for example, natural disasters, cyberattacks, PR crises, pandemics) to minimize damage and ensure business continuity.",
        "keywords": "emergency preparedness planning, disaster recovery services, business continuity planning (Business Continuity Planning), crisis communication strategy, incident response management, reputation recovery, resilience planning"
    },
    {
        "label": "Community Engagement Services",
        "definition": "Facilitating communication and building relationships between organizations (businesses, government, non-profits) and the communities they operate in or serve, often involving outreach, consultation, and partnership building.",
        "keywords": "public outreach programs, stakeholder consultation, community relations, public participation, local partnership development, community needs assessment, social impact initiatives"
    },
    {
        "label": "Stakeholder Services",
        "definition": "Managing relationships and communication with various stakeholders (for example, investors, employees, customers, suppliers, community groups, government) to understand their interests and ensure alignment with organizational goals.",
        "keywords": "stakeholder relationship management, investor relations, employee engagement programs, customer relations, supplier management, government liaison, public affairs"
    },
    {
        "label": "Corporate Responsibility Services",
        "definition": "Assisting businesses in integrating social and environmental concerns into their operations and interactions with stakeholders, often referred to as Corporate Social Responsibility (CSR) or Environmental, Social, and Governance (ESG).",
        "keywords": "Corporate Social Responsibility consulting, Environmental, Social, and Governance reporting, sustainability initiatives, ethical business practices, community investment programs, environmental stewardship, corporate citizenship"
    },
    {
        "label": "Fundraising Services",
        "definition": "Providing strategic advice, planning, and execution support to non-profit organizations or campaigns to solicit donations and raise funds from individuals, corporations, foundations, and government sources.",
        "keywords": "non-profit fundraising consulting, donation campaigns, grant writing services, donor relationship management, capital campaign planning, major gift solicitation, fundraising event management"
    },
    {
        "label": "Volunteer Services",
        "definition": "Recruiting, managing, training, and coordinating volunteers for non-profit organizations, events, or community initiatives.",
        "keywords": "volunteer recruitment, volunteer management programs, non-profit volunteer coordination, event volunteer staffing, community service coordination, volunteer training"
    },
    {
        "label": "Non-Profit Management",
        "definition": "Providing administrative, operational, strategic, and governance support specifically tailored for non-profit organizations, covering areas like fundraising, program management, financial oversight, and board development.",
        "keywords": "non-profit consulting, charity administration, Non-Governmental Organization management support, program evaluation non-profit, board governance consulting, non-profit strategy, capacity building"
    },
    {
        "label": "Arts Services",
        "definition": "Services supporting the creation, presentation, management, and promotion of artistic endeavors, including arts administration, curation, arts marketing, gallery management, and production support for performing arts.",
        "keywords": "arts administration, curatorial services, gallery management, performing arts production, arts marketing, cultural event planning, artist representation"
    },
    {
        "label": "Sports Management Services",
        "definition": "Business aspects of sports and recreation, including managing athletes, teams, events, facilities, marketing, sponsorships, and finances within the sports industry.",
        "keywords": "athlete representation, team operations management, sports event planning, facility management sports, sports marketing agency, sponsorship negotiation, league administration"
    },
    {
        "label": "Fitness Coaching",
        "definition": "Providing personalized guidance, instruction, and motivation to individuals or groups to help them achieve their fitness goals through exercise programs and lifestyle advice.",
        "keywords": "personal training, group fitness instruction, strength coaching, weight loss coaching, athletic performance training, wellness coaching, exercise programming"
    },
    {
        "label": "Health Promotion Services",
        "definition": "Designing and implementing programs and initiatives aimed at enabling people to increase control over and improve their health. This often occurs in workplace, community, or clinical settings.",
        "keywords": "wellness programs, disease prevention initiatives, healthy lifestyle campaigns, workplace wellness, community health education, public health campaigns, behavioral change programs"
    },
    {
        "label": "Physical Therapy Services",
        "definition": "Healthcare services provided by physical therapists to individuals to develop, maintain, and restore maximum movement and functional ability throughout life, often after injury, illness, or surgery.",
        "keywords": "physiotherapy, rehabilitation services, post-surgical rehab, sports injury therapy, manual therapy, therapeutic exercise, mobility improvement"
    },
    {
        "label": "Occupational Health Services",
        "definition": "Healthcare services focused on the health, safety, and well-being of employees in the workplace. This includes pre-employment screenings, injury management, workplace hazard assessment, and wellness programs tailored to work environments.",
        "keywords": "workplace health programs, employee wellness, occupational medicine, work injury treatment, ergonomic assessments workplace, pre-placement physicals, return-to-work programs"
    }
]

# Configuration
XLSX_FILE_PATH = 'insurance_taxonomy.xlsx'
# Attempt to use the first column as the label column if not specified.
# It's better if the user provides the exact column name.
LABEL_COLUMN_NAME = None  # Will be determined from the first column of the loaded Excel.
DEFINITION_HEADER = "Definition"
KEYWORDS_HEADER = "Keywords"

def update_excel_file_pandas(file_path, data_definitions):
    """
    Updates an Excel file with definitions and keywords for specified labels using pandas.
    Adds 'Definition' and 'Keywords' columns if they do not exist.
    """
    global LABEL_COLUMN_NAME # Allow modification of global for determined label column name

    try:
        # Read the Excel file. Pandas uses the first sheet by default.
        # Use engine='openpyxl' for .xlsx files if not automatically inferred or if issues arise.
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Successfully loaded '{file_path}'.")

        if df.empty:
            print("The Excel sheet is empty. Nothing to update.")
            return

        # Determine the label column name if not explicitly set
        if LABEL_COLUMN_NAME is None and len(df.columns) > 0:
            LABEL_COLUMN_NAME = df.columns[0]
            print(f"Using the first column '{LABEL_COLUMN_NAME}' as the label column.")
        elif LABEL_COLUMN_NAME is None:
            print("Error: Could not determine the label column as the Excel sheet has no columns.")
            return
        elif LABEL_COLUMN_NAME not in df.columns:
            print(f"Error: Specified label column '{LABEL_COLUMN_NAME}' not found in the Excel sheet. Found columns: {list(df.columns)}")
            return
            
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as the script.")
        return
    except Exception as e:
        print(f"Error loading Excel file into pandas DataFrame: {e}")
        return

    # Add Definition column if it doesn't exist
    if DEFINITION_HEADER not in df.columns:
        df[DEFINITION_HEADER] = pd.NA  # Initialize with Not Available
        print(f"Added header: '{DEFINITION_HEADER}'")
    else:
        print(f"Found header: '{DEFINITION_HEADER}'")

    # Add Keywords column if it doesn't exist
    if KEYWORDS_HEADER not in df.columns:
        df[KEYWORDS_HEADER] = pd.NA  # Initialize with Not Available
        print(f"Added header: '{KEYWORDS_HEADER}'")
    else:
        print(f"Found header: '{KEYWORDS_HEADER}'")

    updated_count = 0
    not_found_labels = []

    for item in data_definitions:
        label_to_find = item['label']
        definition = item['definition']
        keywords = item['keywords']

        # Find rows where the label column matches
        # Ensure consistent data types for comparison if necessary, e.g. .astype(str)
        match_condition = df[LABEL_COLUMN_NAME].astype(str).str.strip() == str(label_to_find).strip()
        
        if match_condition.any():
            df.loc[match_condition, DEFINITION_HEADER] = definition
            df.loc[match_condition, KEYWORDS_HEADER] = keywords
            print(f"Updated row(s) for label: '{label_to_find}'")
            updated_count += df[match_condition].shape[0]
        else:
            not_found_labels.append(label_to_find)

    if updated_count > 0:
        print(f"Successfully updated {updated_count} entries.")
    else:
        print("No matching labels found in the sheet to update.")

    if not_found_labels:
        print(f"Warning: The following labels were not found in column '{LABEL_COLUMN_NAME}' of the sheet: {not_found_labels}")

    # Save the DataFrame back to Excel
    try:
        # index=False prevents pandas from writing the DataFrame index as a column
        df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"Workbook saved successfully to '{file_path}'")
    except Exception as e:
        print(f"Error saving workbook: {e}")

if __name__ == "__main__":
    print("Starting script to update Excel file using pandas...")
    print("Make sure 'pandas' and 'openpyxl' are installed ('pip install pandas openpyxl').")
    print(f"The script will attempt to modify '{XLSX_FILE_PATH}' in the same directory.")
    print(f"It will try to use the first column for matching labels unless LABEL_COLUMN_NAME is set manually in the script.")
    print("---")
    update_excel_file_pandas(XLSX_FILE_PATH, data_to_add)
    print("---")
    print("Script finished.")
