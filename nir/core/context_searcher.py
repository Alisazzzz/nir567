#All functions for context search are here

def get_context_networkX(query: str, retriever, G):
    search_results = retriever.invoke(query)
    entities = [res.page_content for res in search_results]

    results = {
        "query": query,
        "found_entities": entities,
        "entities_context": [],
        "relationships": [],
    }

    for entity in entities:
            
        entity_data = {
            "entity": entity,
            "node_attributes": dict(G.nodes[entity]),
            "neighbors": []
        }
        
        neighbors = list(G.neighbors(entity))
        
        for neighbor in neighbors:

            neighbor_info = {
                "neighbor": neighbor,
                "neighbor_attributes": dict(G.nodes[neighbor]),
                "edge_attributes": {}
            }
            
            if G.has_edge(entity, neighbor):
                edge_data = G[entity][neighbor]
                neighbor_info["edge_attributes"] = dict(edge_data)
                
                relationship = {
                    "source": entity,
                    "target": neighbor,
                    "attributes": dict(edge_data),
                    "type": edge_data.get('relationship_type', 'connected')
                }
                results["relationships"].append(relationship)
            entity_data["neighbors"].append(neighbor_info)
        results["entities_context"].append(entity_data)

    return results

def prepare_context(context: dict) -> str:
    formatted_lines = []
    for entity_ctx in context['entities_context']:
        entity = entity_ctx['entity']
        formatted_lines.append(f"\n{entity}:")
        
        if entity_ctx['node_attributes'] and entity_ctx['node_attributes'].get('properties') != '{}':
            props = entity_ctx['node_attributes']['properties']
            if props and props != '{}':
                formatted_lines.append(f"  Properties: {props}")
        
        for neighbor_info in entity_ctx['neighbors']:
            neighbor = neighbor_info['neighbor']
            edge_attrs = neighbor_info['edge_attributes']
            
            relation_type = edge_attrs.get('type', 'related_to')
            
            formatted_line = f"  {entity} - {relation_type} - {neighbor}"
            
            if edge_attrs.get('properties') and edge_attrs['properties'] != '{}':
                props = edge_attrs['properties']
                formatted_line += f" [{props}]"
            
            formatted_lines.append(formatted_line)
    
    return "\n".join(formatted_lines)